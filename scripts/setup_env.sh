#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate

pip_install() {
  if python -m pip install "$@"; then
    return 0
  fi

  echo
  echo "Normal pip install failed. Retrying with trusted PyPI hosts..."
  python -m pip install \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host download.pytorch.org \
    "$@"
}

pip_install --upgrade pip
pip_install -r requirements.txt

python - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("mps available:", torch.backends.mps.is_available())
PY

echo
echo "Environment is ready."
echo "Activate it later with:"
echo "source .venv/bin/activate"
