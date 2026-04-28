# vLLM Export Bundle

Minimal, self-contained bundle for launching an OpenAI-compatible vLLM server in Docker and querying it.

## Contents

- `Dockerfile` - lightweight image based on `vllm/vllm-openai`
- `launch_vllm.sh` - starts a vLLM server container
- `query_llm.py` - simple chat-completions query script

## Prerequisites

- Docker
- NVIDIA GPU + NVIDIA Container Toolkit
- (Optional) Hugging Face token for gated models:
  - `export HF_TOKEN=...`

## 1) Build Image

```bash
cd vllm_export
docker build -t vllm-export:latest .
```

You can pin the base image version:

```bash
docker build --build-arg VLLM_BASE_IMAGE=vllm/vllm-openai:v0.8.5 -t vllm-export:latest .
```

## 2) Launch vLLM Server

```bash
cd vllm_export
MODEL_ID=Qwen/Qwen2.5-7B-Instruct-AWQ ./launch_vllm.sh
```

The default model is the official AWQ INT4 build of Qwen2.5-7B-Instruct
(~5 GiB weights), which fits on a single partly-used GPU on this host. To
run the full bf16 model instead (~15 GiB weights, needs a mostly-empty GPU
or TP across multiple), set `MODEL_ID=Qwen/Qwen2.5-7B-Instruct`. vLLM
auto-detects AWQ from the model config; setting `QUANTIZATION=awq` is
optional.

Useful env overrides:

- `MODEL_ID` (default: `Qwen/Qwen2.5-7B-Instruct-AWQ`)
- `IMAGE_NAME` (default: `vllm-export`)
- `TAG` (default: `latest`)
- `HOST_PORT` (default: `8001`)
- `GPU_DEVICES` (`all` or list like `0,1`)
- `TENSOR_PARALLEL_SIZE` (auto-detected from `GPU_DEVICES` if unset)
- `MAX_MODEL_LEN` (default: `8192`)
- `DTYPE` (default: `auto`)
- `GPU_MEMORY_UTILIZATION` (default: `0.5` — tuned for shared GPUs; raise toward `0.9` on a dedicated host for more KV cache)
- `SHM_SIZE` (default: `16g`)
- `QUANTIZATION` (optional, e.g. `awq`)
- `DETACH` (default: `1`) — run the container detached so it survives the launching shell. Set `DETACH=0` to keep the old foreground behavior (logs stream to stdout, but the container dies if the shell exits or the pipeline closes).

When detached, follow logs with `docker logs -f vllm-export-server` and stop the server with `docker rm -f vllm-export-server`.

## 3) Query the Model

```bash
cd vllm_export
python3 query_llm.py \
  --base-url http://127.0.0.1:8001 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --prompt "Summarize vLLM in one sentence."
```

JSON output mode:

```bash
python3 query_llm.py \
  --base-url http://127.0.0.1:8001 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --prompt "Return a JSON object with key answer." \
  --json
```

## Health Checks

```bash
curl http://127.0.0.1:8001/v1/models
```

## Notes

- Container name is `vllm-export-server` by default; existing container with that name is removed on launch.
- Docker publishes only on localhost (`127.0.0.1`) so the API is not exposed on external interfaces by default.
- HF cache is persisted at `../.hf-cache-vllm-export` by default so model downloads are reused.
