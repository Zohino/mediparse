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

mkdir "$work/pdftext"
while IFS= read -r -d '' pdf; do
    relative=${pdf#"$work"/results/eda/}
    pdftotext "$pdf" "$work/pdftext/${relative//\//_}.txt"
done < <(find "$work/results/eda" -name '*.pdf' -print0)

status=0
leaked=$(grep --recursive --files-with-matches --fixed-strings "$prefix-" "$work/eda" "$work/results" "$work/pdftext") || status=$?
case $status in
    0)
        echo "Kanárek: obsah syntetické buňky unikl do výstupu:" >&2
        echo "$leaked" >&2
        exit 1
        ;;
    1) ;;
    *)
        echo "Kanárek selhal: grep skončil s kódem $status." >&2
        exit 2
        ;;
esac
echo "Kanárek nic nenašel. PDF: $work/results/eda"
