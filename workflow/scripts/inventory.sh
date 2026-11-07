#!/usr/bin/env bash
set -euo pipefail

fail() {
    echo "$1" >&2
    exit 1
}

(($# == 3)) || fail "použití: $0 TABULKA OČEKÁVANÝ_SHA256 VÝSTUP"

table=$1
expected_sha256=$2
output=$3

digest=$(sha256sum -- "$table") || fail "$table nejde přečíst"
sha256=${digest%% *}
[[ $sha256 == "$expected_sha256" ]] || fail "$table: otisk SHA-256 nesouhlasí s configem"

# komprimovanost určuje přípona jako v převodu na parquet: gzip, který nejde rozbalit, nesmí projít jako prosté CSV
read_table() {
    case $table in
        *.gz) zcat -- "$table" ;;
        *) cat -- "$table" ;;
    esac
}

# záznam končí řádkem se sudým počtem uvozovek od začátku souboru, zalomení uvnitř pole v uvozovkách ho neukončí
counts=$(read_table | awk '{ quotes += gsub(/"/, "") } quotes % 2 == 0 { n++ } END { print n - 1, quotes % 2 }') || fail "$table nejde přečíst ani rozbalit"
read -r records quotes <<<"$counts"
# hlavička až po úplném přečtení: process substitution chybu rozbalení zahodí a soubor by jen „neměl hlavičku“
read -r header < <(read_table) || fail "$table nemá hlavičku"
((quotes == 0)) || fail "$table má neuzavřené pole v uvozovkách"
((records > 0)) || fail "$table nemá žádný záznam"

jq -n \
    --arg file "${table##*/}" \
    --arg sha256 "$sha256" \
    --argjson bytes "$(stat -c %s -- "$table")" \
    --argjson records "$records" \
    --arg header "${header%$'\r'}" \
    '{file: $file, sha256: $sha256, bytes: $bytes, records: $records, columns: ($header | split(","))}' \
    >"$output"
