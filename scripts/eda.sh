#!/usr/bin/env bash
set -euo pipefail

number=${1:?chybí číslo dokumentu EDA}

# Seznam blokujících proměnných i jeho validace mají jediný zdroj v data_access.json.
reason=$(python -I -c '
import os
import sys
from mediparse.domain.data_access import blocked_reason
from mediparse.infrastructure.data_access_rules import load_data_access_rules
sys.exit(blocked_reason(os.environ, load_data_access_rules()))
' 2>&1) || {
    echo "Reálný render EDA čte data MIMIC; pouští ho jen uživatel ve vlastním terminálu. $reason" >&2
    exit 1
}
root=${PIXI_PROJECT_ROOT:-$(git rev-parse --show-toplevel)}
python -I -c '
import sys
from pathlib import Path
from mediparse.infrastructure.source_revision import GitRevision
sys.exit(GitRevision(Path(sys.argv[1])).read().dirty)
' "$root" || {
    echo "Repo má necommitnutý kód; provenance renderu by ukazovala na jiný kód." >&2
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
if [[ "$number" == 4 ]]; then
    mediparse-eda-note-tokens --parquet "$MEDIPARSE_PARQUET" --translation "$root/translation" --english "$root/resources/synthetic/en" --czech "$root/resources/synthetic/cs" --out "$MEDIPARSE_EDA_CACHE"
fi
work=$(mktemp -d -p "$(dirname "$MEDIPARSE_PARQUET")")
trap 'rm -rf "$work"' EXIT

eda_stage "$root" "$work"
mediparse-eda-provenance --root "$root" --parquet "$MEDIPARSE_PARQUET" --out "$work/provenance.json"
export MEDIPARSE_EDA_PROVENANCE="$work/provenance.json"
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
