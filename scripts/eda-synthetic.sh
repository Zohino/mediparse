#!/usr/bin/env bash
set -euo pipefail

number=${1:-}
prefix=eda-sentinel
root=${PIXI_PROJECT_ROOT:-$(git rev-parse --show-toplevel)}
work=$(mktemp -d)

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/eda-render.sh"

eda_stage "$root" "$work"
mediparse-eda-synthetic-tables --manifest "$root/results/mimic/manifest.json" --out "$work/tables" --prefix "$prefix"

eda_revision "$root"
export MEDIPARSE_PARQUET="$work/tables"

eda_render "$work" "$number"

mkdir "$work/pdftext" "$work/pdfjoined"
while IFS= read -r -d '' pdf; do
    relative=${pdf#"$work"/results/eda/}
    name=${relative//\//_}.txt
    pdftotext "$pdf" "$work/pdftext/$name"
    # Typst dělí dlouhá slova měkkým spojovníkem na konci řádku; token se hledá ve spojeném textu.
    tr -d '\n' <"$work/pdftext/$name" | sed 's/\xc2\xad//g' >"$work/pdfjoined/$name"
done < <(find "$work/results/eda" -name '*.pdf' -print0)

# Buňka nese prefix s pomlčkou. Token textu ze samých písmen (písmena prefixu a řádek, viz
# domain/synthetic_tables.py) přežije tokenizaci, která pomlčky zahodí, a má DF = 1,
# takže se objeví jen ve výstupu, který pustí token jediné zprávy.
token_stem=${prefix//[^[:alpha:]]/}
cell_status=0
token_status=0
cells=$(grep --recursive --files-with-matches --fixed-strings "$prefix-" "$work/eda" "$work/results" "$work/pdftext") || cell_status=$?
tokens=$(grep --recursive --files-with-matches --fixed-strings --ignore-case "$token_stem" "$work/eda" "$work/results" "$work/pdfjoined") || token_status=$?
if ((cell_status > 1 || token_status > 1)); then
    echo "Kanárek selhal: grep skončil s kódem $cell_status a $token_status." >&2
    exit 2
fi
if ((cell_status == 0 || token_status == 0)); then
    echo "Kanárek: obsah syntetické buňky unikl do výstupu:" >&2
    printf '%s\n' "$cells" "$tokens" | grep --invert-match '^$' >&2
    exit 1
fi
echo "Kanárek nic nenašel. PDF: $work/results/eda"
