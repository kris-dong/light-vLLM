#!/usr/bin/env bash
set -euo pipefail

# Launches OpenAI-compatible vLLM server in Docker.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-vllm-export}"
TAG="${TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-export-server}"
CACHE_HOST="${CACHE_HOST:-${ROOT}/../.hf-cache-vllm-export}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
HOST_PORT="${HOST_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
GPU_DEVICES="${GPU_DEVICES:-all}"  # all or list, e.g. 0,1
QUANTIZATION="${QUANTIZATION:-}"   # e.g. awq
DETACH="${DETACH:-1}"              # 1 = run detached (default), 0 = foreground

if [[ "${GPU_DEVICES}" == "all" ]]; then
  DOCKER_GPU_ARG=(--gpus all)
else
  DOCKER_GPU_ARG=(--gpus "device=${GPU_DEVICES}")
fi

if [[ -z "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  if [[ "${GPU_DEVICES}" != "all" ]]; then
    DETECTED_GPU_COUNT="$(awk -F',' '{print NF}' <<<"${GPU_DEVICES}")"
    if [[ "${DETECTED_GPU_COUNT}" =~ ^[0-9]+$ ]] && (( DETECTED_GPU_COUNT > 0 )); then
      TENSOR_PARALLEL_SIZE="${DETECTED_GPU_COUNT}"
    else
      TENSOR_PARALLEL_SIZE=1
    fi
  elif command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${DETECTED_GPU_COUNT}" =~ ^[0-9]+$ ]] && (( DETECTED_GPU_COUNT > 0 )); then
      TENSOR_PARALLEL_SIZE="${DETECTED_GPU_COUNT}"
    else
      TENSOR_PARALLEL_SIZE=1
    fi
  else
    TENSOR_PARALLEL_SIZE=1
  fi
fi

mkdir -p "${CACHE_HOST}"

# Replace existing container for convenience.
if docker ps -a --format '{{.Names}}' | rg -n "^${CONTAINER_NAME}$" >/dev/null; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

DOCKER_ARGS=(
  "${DOCKER_GPU_ARG[@]}"
  --name "${CONTAINER_NAME}"
  --shm-size "${SHM_SIZE:-16g}"
  --rm
  --entrypoint vllm
  -p "127.0.0.1:${HOST_PORT}:8000"
  -v "${CACHE_HOST}:/workspace/.cache/huggingface:rw"
  -e HF_HOME=/workspace/.cache/huggingface
  -e HF_TOKEN="${HF_TOKEN:-}"
  -e HF_HUB_ENABLE_HF_TRANSFER=1
  "${IMAGE_NAME}:${TAG}"
  serve "${MODEL_ID}"
  --host 0.0.0.0
  --port 8000
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --dtype "${DTYPE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

if [[ -n "${QUANTIZATION}" ]]; then
  DOCKER_ARGS+=(--quantization "${QUANTIZATION}")
fi

echo "[start] container=${CONTAINER_NAME} model=${MODEL_ID} bind=127.0.0.1:${HOST_PORT}"

if [[ "${DETACH}" == "1" ]]; then
  docker run -d "${DOCKER_ARGS[@]}" >/dev/null
  cat <<EOF
[started detached]
  follow logs:  docker logs -f ${CONTAINER_NAME}
  health:       curl -fsS http://127.0.0.1:${HOST_PORT}/v1/models
  stop server:  docker rm -f ${CONTAINER_NAME}
EOF
else
  exec docker run "${DOCKER_ARGS[@]}"
fi
