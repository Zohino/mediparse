#!/usr/bin/env bash
set -euo pipefail

max_src_lines=4000
max_test_lines=2000
max_module_lines=300

src_report=$(tokei src/ --output json)
src_lines=$(jq '.Total.code' <<<"$src_report")

if [[ -d tests/ ]]; then
    tests_report=$(tokei tests/ --output json)
    tests_lines=$(jq '.Total.code' <<<"$tests_report")

    tokei tests/

    if ((tests_lines > max_test_lines)); then
        echo "VAROVÁNÍ: tests/ má $tests_lines LOC (limit: $max_test_lines)"
    fi
fi

tokei src/

if ((src_lines > max_src_lines)); then
    echo "VAROVÁNÍ: src/ má $src_lines LOC (limit: $max_src_lines)"
fi

jq -r --argjson limit "$max_module_lines" '
    .Python.reports[]
    | select(.stats.code > $limit)
    | "VAROVÁNÍ: \(.name) má \(.stats.code) LOC (limit: \($limit))"
' <<<"$src_report"
