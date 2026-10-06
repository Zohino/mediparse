#!/bin/bash
#PBS -N mediparse-translate
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_mem=40gb:gpu_cap=compute_80:scratch_ssd=80gb
#PBS -l walltime=2:00:00
set -euo pipefail

workdir=${PBS_O_WORKDIR:-$PWD}
cache=${CACHE_DIR:-${SCRATCHDIR:?CACHE_DIR ani SCRATCHDIR nejsou nastavené}}

if [[ ${PBS_ENVIRONMENT:-} == PBS_BATCH ]]; then
    trap 'clean_scratch' EXIT TERM
fi

export PATH="$workdir/bin:$PATH"
export HF_HOME="$cache/huggingface"
export UV_CACHE_DIR="$cache/uv"
export UV_PYTHON_INSTALL_DIR="$cache/python"
HF_TOKEN=$(<"$workdir/.hf_token")
export HF_TOKEN

args=(--workdir "$workdir")
if [[ -n ${IDS:-} ]]; then
    read -ra ids <<<"$IDS"
    args+=(--ids "${ids[@]}")
fi

nvidia-smi
uv --version
time uv run --script --locked "$workdir/translate.py" "${args[@]}"
