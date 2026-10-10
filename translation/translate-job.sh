#!/bin/bash
#PBS -N mediparse-translate
#PBS -l select=1:ncpus=8:mem=128gb:ngpus=1:gpu_mem=80gb:gpu_cap=compute_80:scratch_ssd=200gb
#PBS -l walltime=2:00:00
set -euo pipefail

workdir=${PBS_O_WORKDIR:-$PWD}
cache=${CACHE_DIR:-${SCRATCHDIR:?CACHE_DIR ani SCRATCHDIR nejsou nastavené}}

if [[ ${PBS_ENVIRONMENT:-} == PBS_BATCH ]]; then
    trap 'clean_scratch' EXIT TERM
fi

export PATH="$workdir/bin:$PATH"
export HF_HOME="$cache/huggingface"
export PIXI_HOME="$cache/pixi-home"
export PIXI_CACHE_DIR="$cache/pixi"
mkdir -p "$PIXI_HOME"
printf 'detached-environments = "%s"\n' "$cache/envs" >"$PIXI_HOME/config.toml"
HF_TOKEN=$(<"$workdir/.hf_token")
export HF_TOKEN

args=(--workdir "$workdir")
if [[ -n ${IDS:-} ]]; then
    read -ra ids <<<"$IDS"
    args+=(--ids "${ids[@]}")
fi

nvidia-smi
pixi --version
time pixi run --manifest-path "$workdir/pyproject.toml" --frozen -e translate python "$workdir/translate.py" "${args[@]}"
