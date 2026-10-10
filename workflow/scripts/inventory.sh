#!/usr/bin/env bash
set -euo pipefail

fail() {
    echo "$1" >&2
    exit 1
}

(($# == 2)) || fail "použití: $0 TABULKA VÝSTUP"

table=$1
output=$2

read -r header < <(zcat -- "$table") || fail "$table nemá hlavičku"
# záznam končí řádkem se sudým počtem uvozovek od začátku souboru, zalomení uvnitř pole v uvozovkách ho neukončí
counts=$(zcat -- "$table" | awk '{ quotes += gsub(/"/, "") } quotes % 2 == 0 { n++ } END { print n - 1, quotes % 2 }') || fail "$table nejde rozbalit"
read -r records quotes <<<"$counts"
((quotes == 0)) || fail "$table má neuzavřené pole v uvozovkách"
((records > 0)) || fail "$table nemá žádný záznam"

jq -n \
    --arg file "${table##*/}" \
    --argjson bytes "$(stat -c %s -- "$table")" \
    --argjson records "$records" \
    --arg header "${header%$'\r'}" \
    '{file: $file, bytes: $bytes, records: $records, columns: ($header | split(","))}' \
    >"$output"
