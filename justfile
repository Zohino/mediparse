# spust pres uv
runner := "uv run"

max-src-lines := "4000"
max-test-lines := "2000"
max-module-lines := "300"
max-scripts-lines := "20"

# výpis všech dostupných příkazů
default:
    @just --list

# commit pomocí commitizen
commit:
    {{runner}} cz commit

# statistiky kódu pomocí tokei - varování proti overengineeringu
count-lines:
    #!/usr/bin/env bash
    set -euo pipefail

    src_report=$(tokei src/ --output json)
    src_lines=$(jq '.Total.code' <<< "$src_report")

    if [[ -d tests/ ]]; then
        tests_report=$(tokei tests/ --output json)
        tests_lines=$(jq '.Total.code' <<< "$tests_report")

        tokei tests/

        if (( tests_lines > {{max-test-lines}} )); then
            echo "VAROVÁNÍ: tests/ má $tests_lines LOC (limit: {{max-test-lines}})"
        fi
    fi

    tokei src/

    if (( src_lines > {{max-src-lines}} )); then
        echo "VAROVÁNÍ: src/ má $src_lines LOC (limit: {{max-src-lines}})"
    fi


    jq -r '
        .Python.reports[]
        | select(.stats.code > {{max-module-lines}})
        | "VAROVÁNÍ: \(.name) má \(.stats.code) LOC (limit: {{max-module-lines}})"
    ' <<< "$src_report"

# kontrola zakázaných importů
lint-imports:
    #!/usr/bin/env bash
    set -uo pipefail

    failed=0

    if [[ -d workflow/scripts/ ]]; then
        scripts_report=$(tokei workflow/scripts/ --output json)

        warnings=$(jq -r '
            (.Python.reports // [])[]
            | select(.stats.code > {{max-scripts-lines}})
            | "VAROVÁNÍ: \(.name) má \(.stats.code) LOC (limit: {{max-scripts-lines}})"
        ' <<< "$scripts_report")

        if [[ -n "$warnings" ]]; then
            echo "$warnings"
            failed=1
        fi
    fi

    if ! {{runner}} lint-imports; then
        failed=1
    fi

    exit "$failed"
