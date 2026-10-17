#!/usr/bin/env bash
set -euo pipefail

base=${1:-origin/main}

changed=$(git diff --name-only --diff-filter=AMR "$base...HEAD" -- 'src/*.py' | paste -sd, -)
if [[ -z $changed ]]; then
    echo "Větev proti $base nemění žádný modul v src/."
    exit 0
fi

# Průběh testů jde na stderr, aby stdout nesl jen report a dal se připojit do podkladu reviewera.
pytest -q --cov --cov-report= >&2
coverage report --show-missing --include="$changed"
