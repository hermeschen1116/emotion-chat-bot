#!/usr/bin/zsh
pip_path="$HOME/.pyenv/shims/pip"
${pip_path} install -U pip wheel setuptools pip-review
${pip_path} install -U jupyter
${pip_path} install -U polars datasets tensorboard
${pip_path} install -U fasttext
${pip_path} install -U https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip
case $(uname) in
    "Darwin")
        ${pip_path} install -U --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu
        ;;
    "Linux")
        ${pip_path} install -U --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cu118
        ;;
esac
