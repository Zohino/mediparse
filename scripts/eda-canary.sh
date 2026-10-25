# shellcheck shell=bash

# Kanárek renderu nad syntetikou: hledá obsah syntetických buněk v kopii eda/ (včetně freeze),
# ve výstupu results/ a v textu PDF (pdftext/). Vrací 0 bez nálezu, 1 při úniku, 2 při chybě.
#
# Buňka nese prefix s pomlčkou; provenance uvádí prefix bez pomlčky za ním, proto se hledá
# s ní. Token textu ze samých písmen (písmena prefixu a řádek, viz domain/synthetic_tables.py)
# přežije tokenizaci, která pomlčky zahodí, a má DF = 1, takže se objeví jen ve výstupu,
# který pustí token jediné zprávy.
eda_canary() {
    local work=$1 prefix=$2
    local stem=${prefix//[^[:alpha:]]/} text cells tokens cell_status=0 token_status=0
    mkdir -p "$work/pdfjoined"
    for text in "$work"/pdftext/*.txt; do
        [[ -f $text ]] || continue
        # Typst dělí dlouhá slova měkkým spojovníkem na konci řádku; token se hledá ve spojeném textu.
        tr -d '\n' <"$text" | sed 's/\xc2\xad//g' >"$work/pdfjoined/${text##*/}"
    done
    cells=$(grep --recursive --files-with-matches --fixed-strings "$prefix-" "$work/eda" "$work/results" "$work/pdftext") || cell_status=$?
    tokens=$(grep --recursive --files-with-matches --fixed-strings --ignore-case "$stem" "$work/eda" "$work/results" "$work/pdfjoined") || token_status=$?
    if ((cell_status > 1 || token_status > 1)); then
        echo "Kanárek selhal: grep skončil s kódem $cell_status a $token_status." >&2
        return 2
    fi
    if ((cell_status == 0 || token_status == 0)); then
        echo "Kanárek: obsah syntetické buňky unikl do výstupu:" >&2
        printf '%s\n' "$cells" "$tokens" | grep --invert-match '^$' >&2
        return 1
    fi
}
