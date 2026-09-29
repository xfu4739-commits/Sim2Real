#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT="${1:-${SIM2REAL_DATASET_ROOT:-${HOME}/Datasets/Sim2Real-Fire}}"
LEVEL="${2:-full}"
CONFIG="${3:-${REPO_ROOT}/config/config.yaml}"
REPORT="${4:-${REPO_ROOT}/output/dataset_validation_linux.json}"

case "${LEVEL}" in
  structure|sample|full) ;;
  *)
    echo "级别必须是 structure、sample 或 full" >&2
    exit 2
    ;;
esac

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "未找到 Python。请先安装项目 requirements.txt 中的依赖。" >&2
  exit 2
fi

"${PYTHON}" "${SCRIPT_DIR}/validate_dataset.py" \
  --dataset-root "${DATASET_ROOT}" \
  --config "${CONFIG}" \
  --level "${LEVEL}" \
  --report "${REPORT}"
exit $?
