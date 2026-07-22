#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR=${DQN_VENV_DIR:-"$REPO_ROOT/.venv"}
PYTHON_BIN=${DQN_PYTHON:-"$VENV_DIR/bin/python"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  printf '%s\n' \
    "Usage: scripts/reproduce.sh setup|test|smoke|verify-reference|eval|full [arguments]" \
    "" \
    "  setup                         Create .venv, install dependencies and AutoROM." \
    "  test                          Run the unit test suite." \
    "  smoke [output-dir]            Run and verify the 256-decision CPU smoke." \
    "  verify-reference              Verify committed EXP-0004 hashes and headline values." \
    "  eval <checkpoint> [output]    Run the 135K-decision checkpoint evaluation." \
    "  full [output-dir]             Run the frozen 10M-decision CUDA reproduction."
}

require_python() {
  if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python environment not found: %s\nRun DQN_ACCEPT_ROM_LICENSE=1 scripts/reproduce.sh setup first.\n' "$PYTHON_BIN" >&2
    exit 2
  fi
}

command_name=${1:-}
case "$command_name" in
  setup)
    if [[ ${DQN_ACCEPT_ROM_LICENSE:-0} != 1 ]]; then
      printf '%s\n' \
        "Atari ROMs have a separate license and are not distributed by this repository." \
        "Review that license, then rerun with DQN_ACCEPT_ROM_LICENSE=1." >&2
      exit 2
    fi
    bootstrap_python=${DQN_BOOTSTRAP_PYTHON:-python3}
    "$bootstrap_python" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools
    "$VENV_DIR/bin/python" -m pip install -e "$REPO_ROOT[atari]"
    "$VENV_DIR/bin/AutoROM" --accept-license --quiet
    "$VENV_DIR/bin/python" -c "import ale_py, gymnasium, torch; print('setup ok:', torch.__version__)"
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
      --device "${DQN_EVAL_DEVICE:-cuda}"
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

