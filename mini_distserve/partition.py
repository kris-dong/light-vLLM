"""
Model partitioning descriptor.

We don't actually shard weights here — that's what real frameworks (Megatron,
DeepSpeed, vLLM) do. What this module *does* do at runtime:

  * Read the model's ``config.json`` from a HuggingFace snapshot dir.
  * Compute KV-bytes-per-token from (num_kv_heads, head_dim, num_layers, dtype).
  * Given a per-GPU memory budget and a TP/PP layout, compute how many KV
    blocks fit on each engine.
  * Provide a ``ParallelLayout`` the engine uses to know how many GPUs it owns
    and which CUDA devices to bind to.

This is "the partition decision at runtime": you tell it the layout and the
budget, and it returns the sizing the rest of the system uses.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Bytes per element by dtype name.
_DTYPE_BYTES: Dict[str, int] = {
    "fp32": 4, "float32": 4,
    "fp16": 2, "float16": 2, "half": 2,
    "bf16": 2, "bfloat16": 2,
    "fp8": 1, "float8": 1,
    "int8": 1,
}


def _resolve_snapshot_dir(model_path: str) -> Path:
    """
    Accepts either a snapshot directory (containing config.json) or a HF cache
    directory of the form ``models--Qwen--Qwen2.5-7B-Instruct``. In the latter
    case we follow ``snapshots/<sha>/`` to the actual files.
    """
    p = Path(model_path)
    if (p / "config.json").exists():
        return p
    snap_root = p / "snapshots"
    if snap_root.is_dir():
        snaps = [d for d in snap_root.iterdir() if d.is_dir()]
        if snaps:
            # Prefer the snapshot that actually has a config.json.
            for d in snaps:
                if (d / "config.json").exists():
                    return d
            return snaps[0]
    raise FileNotFoundError(f"no config.json found under {model_path!r}")


@dataclass
class ParallelLayout:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    # CUDA device ids assigned to this engine instance; len == TP * PP.
    device_ids: Tuple[int, ...] = (0,)

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    def __post_init__(self) -> None:
        if self.world_size != len(self.device_ids):
            raise ValueError(
                f"device_ids has {len(self.device_ids)} entries but world_size="
                f"{self.world_size} (TP={self.tensor_parallel_size},"
                f"PP={self.pipeline_parallel_size})"
            )


@dataclass
class ModelPartition:
    """Sized partition for one engine instance."""

    model_name: str
    model_path: str
    layout: ParallelLayout

    # Architecture, read from HF config.
    num_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    max_position_embeddings: int
    dtype: str

    # Computed.
    kv_bytes_per_token_per_layer: int
    kv_bytes_per_token: int          # across all layers, after TP/PP shard
    block_size_tokens: int
    kv_block_bytes: int
    total_kv_blocks: int             # per engine (sum across this engine's GPUs)

    # Quantization (None means dense / unquantized weights).
    quant_method: Optional[str] = None
    quant_bits: Optional[int] = None
    # Estimated bytes the model weights occupy on the engine after TP/PP sharding.
    weight_bytes_total: int = 0

    @classmethod
    def from_hf_config(
        cls,
        model_path: str,
        layout: ParallelLayout,
        per_gpu_kv_budget_bytes: int,
        block_size_tokens: int = 16,
        dtype_override: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> "ModelPartition":
        snap = _resolve_snapshot_dir(model_path)
        cfg = json.loads((snap / "config.json").read_text())

        num_layers = int(cfg.get("num_hidden_layers", cfg.get("n_layer", 0)))
        num_attn_heads = int(cfg.get("num_attention_heads", cfg.get("n_head", 0)))
        num_kv_heads = int(cfg.get("num_key_value_heads", num_attn_heads))
        hidden_size = int(cfg.get("hidden_size", cfg.get("d_model", 0)))
        head_dim = int(cfg.get("head_dim", hidden_size // max(1, num_attn_heads)))
        max_pos = int(cfg.get("max_position_embeddings", 32768))
        dtype = (dtype_override or cfg.get("torch_dtype", "bf16")).lower()

        quant_cfg = cfg.get("quantization_config") or {}
        quant_method = quant_cfg.get("quant_method") if quant_cfg else None
        quant_bits = int(quant_cfg["bits"]) if quant_cfg.get("bits") is not None else None

        if not all([num_layers, num_kv_heads, head_dim]):
            raise ValueError(
                f"could not derive shape from config.json at {snap}: "
                f"layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}"
            )

        # KV cache stays in compute dtype (fp16/bf16) even for AWQ/GPTQ weights —
        # quantization only affects weights, not the K/V tensors produced at runtime.
        kv_compute_dtype = dtype
        elem_bytes = _DTYPE_BYTES.get(kv_compute_dtype, 2)
        # KV = K + V; both same shape. Per token, per layer, per (kv_head * head_dim).
        # Under TP, kv_heads shard across TP ranks.
        kv_heads_per_rank = max(1, num_kv_heads // layout.tensor_parallel_size)
        kv_bpt_per_layer = 2 * kv_heads_per_rank * head_dim * elem_bytes
        # PP shards layers; each engine owns layers/PP of them.
        layers_per_engine = max(1, num_layers // layout.pipeline_parallel_size)
        kv_bpt = kv_bpt_per_layer * layers_per_engine

        kv_block_bytes = kv_bpt * block_size_tokens
        # Per-GPU budget × number of GPUs this engine owns.
        total_budget = per_gpu_kv_budget_bytes * layout.world_size
        total_kv_blocks = max(1, total_budget // kv_block_bytes)

        weight_bytes_total = _estimate_weight_bytes(
            cfg=cfg,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_attn_heads=num_attn_heads,
            dtype_bytes=_DTYPE_BYTES.get(dtype, 2),
            quant_method=quant_method,
            quant_bits=quant_bits,
            quant_group_size=int(quant_cfg.get("group_size", 128)) if quant_cfg else 128,
            layers_per_engine=layers_per_engine,
        )

        return cls(
            model_name=model_name or os.path.basename(str(snap)),
            model_path=str(snap),
            layout=layout,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            max_position_embeddings=max_pos,
            dtype=dtype,
            kv_bytes_per_token_per_layer=kv_bpt_per_layer,
            kv_bytes_per_token=kv_bpt,
            block_size_tokens=block_size_tokens,
            kv_block_bytes=kv_block_bytes,
            total_kv_blocks=int(total_kv_blocks),
            quant_method=quant_method,
            quant_bits=quant_bits,
            weight_bytes_total=int(weight_bytes_total),
        )

    def summary(self) -> str:
        q = (
            f"{self.quant_method}-{self.quant_bits}bit"
            if self.quant_method else f"dense-{self.dtype}"
        )
        return (
            f"{self.model_name}  TP={self.layout.tensor_parallel_size} "
            f"PP={self.layout.pipeline_parallel_size} devices={self.layout.device_ids}\n"
            f"  layers={self.num_layers} kv_heads={self.num_kv_heads} "
            f"head_dim={self.head_dim} dtype={self.dtype}  weights={q}\n"
            f"  weight bytes (this engine) ~= {self.weight_bytes_total:,} "
            f"({self.weight_bytes_total / (1024**3):.2f} GiB)\n"
            f"  KV bytes/token (this engine, after TP/PP) = "
            f"{self.kv_bytes_per_token:,}\n"
            f"  block_size={self.block_size_tokens} tokens "
            f"-> {self.kv_block_bytes:,} bytes/block\n"
            f"  total KV blocks on engine = {self.total_kv_blocks}"
        )


def _estimate_weight_bytes(
    *,
    cfg: Dict,
    num_layers: int,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    num_attn_heads: int,
    dtype_bytes: int,
    quant_method: Optional[str],
    quant_bits: Optional[int],
    quant_group_size: int,
    layers_per_engine: int,
) -> int:
    """
    Rough per-engine weight footprint. Counts attention QKV+O and MLP up/gate/down,
    optionally quantized. Embeddings/norms/lm_head are added once for the engine
    that owns the input/output stage; here we just include them globally — under
    PP this slightly over-estimates the per-engine weight, which is fine as a
    capacity-planning lower bound on free KV memory.
    """
    intermediate = int(cfg.get("intermediate_size", 4 * hidden_size))
    vocab = int(cfg.get("vocab_size", 0))

    # Per-layer linear param counts (number of weight elements).
    qkv_in = hidden_size
    qkv_out = (num_attn_heads + 2 * num_kv_heads) * head_dim
    o_in = num_attn_heads * head_dim
    o_out = hidden_size
    mlp_a = hidden_size * intermediate           # gate or up
    mlp_b = intermediate * hidden_size           # down
    per_layer_linear_elems = (
        qkv_in * qkv_out  # fused qkv (or 3 separate; same total)
        + o_in * o_out
        + 2 * mlp_a       # gate + up
        + mlp_b
    )

    if quant_method in ("awq", "gptq") and quant_bits:
        # 4-bit weights packed; group-wise scales+zeros add ~2 fp16 elems per group.
        wbytes = per_layer_linear_elems * quant_bits / 8.0
        groups = per_layer_linear_elems / max(1, quant_group_size)
        wbytes += groups * 2 * 2  # scale + zero in fp16 (2 bytes each)
        per_layer_bytes = wbytes
    else:
        per_layer_bytes = per_layer_linear_elems * dtype_bytes

    # Embedding + lm_head are dense in fp16/bf16 even with quantized linears.
    emb_bytes = vocab * hidden_size * dtype_bytes * 2  # tied or untied: count both
    norm_bytes = num_layers * 2 * hidden_size * dtype_bytes  # 2 RMSNorms per layer

    return int(per_layer_bytes * layers_per_engine + emb_bytes + norm_bytes)
