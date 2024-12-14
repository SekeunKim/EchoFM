#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.14 -y
    conda activate $CONDA_ENV

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

pip install iopath, psutil, scipy, einops, tensorboard
conda install simplejson
# # This is required to enable PEP 660 support
# pip install --upgrade pip setuptools

# # Install FlashAttention2
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# # Install VILA
# pip install -e ".[train,eval]"

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

# pip install git+https://github.com/huggingface/transformers@v4.36.2
