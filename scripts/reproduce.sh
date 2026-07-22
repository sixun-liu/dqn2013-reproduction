#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
command_name=${1:-}
if [[ -n ${DQN_VENV_DIR:-} ]]; then
  VENV_DIR=$DQN_VENV_DIR
elif [[ $command_name == "setup-gpu" || $command_name == "full" ]]; then
  VENV_DIR="$REPO_ROOT/.venv-gpu"
else
  VENV_DIR="$REPO_ROOT/.venv"
fi
PYTHON_BIN=${DQN_PYTHON:-"$VENV_DIR/bin/python"}
DQN_CPU_TORCH_INDEX_URL=${DQN_CPU_TORCH_INDEX_URL:-"https://download.pytorch.org/whl/cpu"}
DQN_TORCH_SPEC=${DQN_TORCH_SPEC:-"torch>=2.3,<3"}
if [[ ! ${OMP_NUM_THREADS:-} =~ ^[1-9][0-9]*$ ]]; then
  OMP_NUM_THREADS=1
fi
if [[ ! ${MKL_NUM_THREADS:-} =~ ^[1-9][0-9]*$ ]]; then
  MKL_NUM_THREADS=1
fi
export OMP_NUM_THREADS MKL_NUM_THREADS
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  printf '%s\n' \
    "Usage: scripts/reproduce.sh setup|setup-gpu|test|smoke|verify-reference|eval|full [arguments]" \
    "" \
    "  setup                         Create a lightweight CPU .venv for verification." \
    "  setup-gpu                     Create .venv-gpu using an explicit CUDA wheel index." \
    "  test                          Run the unit test suite." \
    "  smoke [output-dir]            Run and verify the 256-decision CPU smoke." \
    "  verify-reference              Verify committed EXP-0004 hashes and headline values." \
    "  eval <checkpoint> [output]    Run the 135K-decision checkpoint evaluation on CPU." \
    "  full [output-dir]             Run the frozen 10M-decision CUDA reproduction."
}

require_python() {
  if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    setup_command=setup
    if [[ $command_name == "full" ]]; then
      setup_command=setup-gpu
    fi
    printf 'Python environment not found: %s\nRun the documented %s command first.\n' \
      "$PYTHON_BIN" "$setup_command" >&2
    exit 2
  fi
}

require_rom_acceptance() {
  if [[ ${DQN_ACCEPT_ROM_LICENSE:-0} != 1 ]]; then
    printf '%s\n' \
      "Atari ROMs have a separate license and are not distributed by this repository." \
      "Review that license, then rerun with DQN_ACCEPT_ROM_LICENSE=1." >&2
    exit 2
  fi
}

create_environment() {
  local torch_index_url=$1
  local expected_device=$2
  local bootstrap_python=${DQN_BOOTSTRAP_PYTHON:-python3}

  "$bootstrap_python" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools
  "$VENV_DIR/bin/python" -m pip install --no-cache-dir "$DQN_TORCH_SPEC" \
    --index-url "$torch_index_url"
  "$VENV_DIR/bin/python" -m pip install --no-cache-dir -e "$REPO_ROOT[runtime,atari]"
  "$VENV_DIR/bin/AutoROM" --accept-license --quiet
  "$VENV_DIR/bin/python" - "$expected_device" <<'PY'
import sys

import ale_py
import gymnasium
import torch

expected_device = sys.argv[1]
if expected_device == "cuda" and not torch.cuda.is_available():
    raise SystemExit(
        "A CUDA PyTorch wheel was installed, but torch.cuda.is_available() is false. "
        "Check the selected wheel index and host driver."
    )
print("setup ok:", torch.__version__, "cuda_available=", torch.cuda.is_available())
PY
}

case "$command_name" in
  setup)
    require_rom_acceptance
    create_environment "$DQN_CPU_TORCH_INDEX_URL" cpu
    ;;
  setup-gpu)
    require_rom_acceptance
    if [[ -z ${DQN_TORCH_INDEX_URL:-} ]]; then
      printf '%s\n' \
        "GPU setup is intentionally explicit because CUDA wheels are large and hardware-specific." \
        "Set DQN_TORCH_INDEX_URL to the wheel index selected for this host in the PyTorch installer." >&2
      exit 2
    fi
    create_environment "$DQN_TORCH_INDEX_URL" cuda
    ;;
  test)
    require_python
    "$PYTHON_BIN" -m unittest discover -s "$REPO_ROOT/tests" -v
    ;;
  smoke)
    require_python
    output_dir=${2:-"$REPO_ROOT/runs/nature2015-smoke-$(date -u +%Y%m%dT%H%M%SZ)"}
    "$PYTHON_BIN" "$REPO_ROOT/scripts/run_nature2015_config.py" \
      --config "$REPO_ROOT/configs/public/nature2015_smoke_cpu.json" \
      --output-dir "$output_dir" \
      --device cpu
    "$PYTHON_BIN" "$REPO_ROOT/scripts/verify_run.py" --run-dir "$output_dir" --mode smoke
    ;;
  verify-reference)
    require_python
    "$PYTHON_BIN" "$REPO_ROOT/scripts/verify_reference.py"
    ;;
  eval)
    require_python
    checkpoint=${2:?checkpoint path is required}
    output=${3:-"$REPO_ROOT/runs/checkpoint-eval-$(date -u +%Y%m%dT%H%M%SZ).json"}
    "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_dqn2015_checkpoint.py" \
      --checkpoint "$checkpoint" \
      --output "$output" \
      --device "${DQN_EVAL_DEVICE:-cpu}"
    ;;
  full)
    require_python
    output_dir=${2:-"$REPO_ROOT/runs/nature2015-table3-$(date -u +%Y%m%dT%H%M%SZ)"}
    "$PYTHON_BIN" "$REPO_ROOT/scripts/run_nature2015_config.py" \
      --config "$REPO_ROOT/configs/public/nature2015_table3_10m.json" \
      --output-dir "$output_dir" \
      --device cuda
    "$PYTHON_BIN" "$REPO_ROOT/scripts/verify_run.py" --run-dir "$output_dir" --mode full
    ;;
  *)
    usage
    exit 2
    ;;
esac
