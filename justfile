# spust pres uv
runner := "uv run"

max-src-lines := "4000"
max-test-lines := "2000"
max-module-lines := "300"


# výpis všech dostupných příkazů
default:
    @just --list

# commit pomocí commitizen
commit:
    {{runner}} cz commit

# statistiky kódu pomocí tokei - varování proti overengineeringu
check-lines:
    #!/usr/bin/env bash
    set -euo pipefail

    src_report=$(tokei src/ --output json)
    tests_report=$(tokei tests/ --output json)

    src_lines=$(jq '.Total.code' <<< "$src_report")
    tests_lines=$(jq '.Total.code' <<< "$tests_report")

    tokei src/ tests/

    if (( src_lines > {{max-src-lines}} )); then
        echo "VAROVÁNÍ: src/ má $src_lines LOC (limit: {{max-src-lines}})"
    fi

    if (( tests_lines > {{max-test-lines}} )); then
        echo "VAROVÁNÍ: tests/ má $tests_lines LOC (limit: {{max-test-lines}})"
    fi

    jq -r '
        .Python.reports[]
        | select(.stats.code > {{max-module-lines}})
        | "VAROVÁNÍ: \(.name) má \(.stats.code) LOC (limit: {{max-module-lines}})"
    ' <<< "$src_report"


