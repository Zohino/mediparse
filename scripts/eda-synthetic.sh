#!/usr/bin/env bash
set -euo pipefail

number=${1:-}
prefix=eda-sentinel
root=${PIXI_PROJECT_ROOT:-$(git rev-parse --show-toplevel)}
work=$(mktemp -d)

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/eda-render.sh"
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/eda-canary.sh"

eda_stage "$root" "$work"
mediparse-eda-synthetic-tables --manifest "$root/results/mimic/manifest.json" --candidates "$root/config/diagnosis_candidates.json" --corpus-rule "$root/config/input_corpus.json" --lexicon "$root/config/section_headers.json" --plans "$root/resources/synthetic/plans.jsonl" --corpus "$root/resources/synthetic" --out "$work/tables" --prefix "$prefix"

eda_revision "$root"
export MEDIPARSE_PARQUET="$work/tables"
export MEDIPARSE_EDA_CACHE="$work/eda-cache"
mediparse-eda-provenance --root "$root" --parquet "$MEDIPARSE_PARQUET" --out "$work/provenance.json"
export MEDIPARSE_EDA_PROVENANCE="$work/provenance.json"
mediparse-eda-note-features --parquet "$MEDIPARSE_PARQUET" --lexicon "$root/config/section_headers.json" --candidates "$root/config/diagnosis_candidates.json" --variants "$root/config/diagnosis_variants.json" --out "$MEDIPARSE_EDA_CACHE"

mediparse-eda-note-tokens --parquet "$MEDIPARSE_PARQUET" --translation "$root/translation" --english "$root/resources/synthetic/en" --czech "$root/resources/synthetic/cs" --out "$MEDIPARSE_EDA_CACHE"

eda_render "$work" "$number"

mkdir "$work/pdftext"
while IFS= read -r -d '' pdf; do
    relative=${pdf#"$work"/results/eda/}
    pdftotext "$pdf" "$work/pdftext/${relative//\//_}.txt"
done < <(find "$work/results/eda" -name '*.pdf' -print0)

eda_canary "$work" "$prefix"
echo "Kanárek nic nenašel. PDF: $work/results/eda"
