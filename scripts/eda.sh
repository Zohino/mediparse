#!/usr/bin/env bash
set -euo pipefail

number=${1:?chybí číslo dokumentu EDA}
root=${PIXI_PROJECT_ROOT:-$(git rev-parse --show-toplevel)}

[[ -z "${CLAUDECODE:-}" ]] || {
    echo "Reálný render EDA čte data MIMIC; pouští ho jen uživatel ve vlastním terminálu, ne Claude Code." >&2
    exit 1
}
[[ -z "$(git -C "$root" status --porcelain)" ]] || {
    echo "Repo má necommitnuté změny; provenance renderu by ukazovala na jiný kód." >&2
    exit 1
}

documents=()
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/eda-render.sh"

export MEDIPARSE_PARQUET=${MEDIPARSE_PARQUET:-$root/resources/mimic/parquet}
work=$(mktemp -d -p "$(dirname "$MEDIPARSE_PARQUET")")
trap 'rm -rf "$work"' EXIT

eda_stage "$root" "$work"
eda_revision "$root"
eda_render "$work" "$number"

for document in "${documents[@]}"; do
    stem=$(basename "$document" .qmd)
    rm -rf "${root:?}/eda/_freeze/$stem"
    mkdir -p "$root/eda/_freeze" "$root/results/eda"
    cp -r "$work/eda/_freeze/$stem" "$root/eda/_freeze/$stem"
    cp "$work/results/eda/$stem.pdf" "$root/results/eda/$stem.pdf"
done
