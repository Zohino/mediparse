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
export MEDIPARSE_EDA_CACHE=${MEDIPARSE_EDA_CACHE:-$root/resources/mimic/eda-cache}
# Výpočet nad celou discharge stojí minuty a potřebuje ho jen sešit 9.
if [[ "$number" == 9 ]]; then
    mediparse-eda-note-features --parquet "$MEDIPARSE_PARQUET" --lexicon "$root/config/section_headers.json" --candidates "$root/config/diagnosis_candidates.json" --out "$MEDIPARSE_EDA_CACHE"
fi
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
    rm -rf "${root:?}/results/eda/$stem"
    if [[ -d "$work/results/eda/$stem" ]]; then
        cp -r "$work/results/eda/$stem" "$root/results/eda/$stem"
    fi
done
