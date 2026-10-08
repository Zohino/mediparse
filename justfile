# spust pres uv
runner := "uv run"
container-engine := env("CONTAINER_ENGINE", `command -v podman || command -v docker || echo docker`)

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

# Linting pomocí ruff
lint:
    {{runner}} ruff check

# Formátovat kód pomocí ruff
format:
    {{runner}} ruff format

# Kontrola statického typování pomocí ty
typecheck:
    {{runner}} ty check

# Statistika lintingu pomocí ruff pro vyloučení omezujících pravidel
ruff-statistics:
    {{runner}} ruff check --statistics --no-fix

# audit syntetického korpusu proti MIMIC-IV-Note; jen lokálně, mimo CI a Claude Code, s commitnutým repem mimo korpus
audit-corpus discharge radiology report:
    #!/usr/bin/env bash
    set -euo pipefail

    [[ -z "$(git status --porcelain -- . ':!resources/synthetic')" ]] || { echo "Repo má mimo korpus necommitnuté změny; záznam auditu by ukazoval na jiný kód." >&2; exit 1; }
    {{runner}} mediparse-corpus-audit --reference "{{discharge}}" --reference "{{radiology}}" --report "{{report}}" --commit "$(git rev-parse HEAD)"

# provenance auditovaného korpusu; commit specifikace z historie docs/synthetic-corpus.md
provenance-corpus model:
    #!/usr/bin/env bash
    set -euo pipefail

    [[ -z "$(git status --porcelain -- docs/synthetic-corpus.md)" ]] || { echo "Specifikace má necommitnuté změny." >&2; exit 1; }
    spec=$(git log -1 --format=%H -- docs/synthetic-corpus.md)
    git merge-base --is-ancestor "$spec" origin/main || { echo "Commit specifikace $spec není v origin/main; rebase merge by ho přepsal, změnu specifikace merguj dřív." >&2; exit 1; }
    {{runner}} mediparse-corpus-provenance --model "{{model}}" --claude-code-version "$(claude --version | cut -d ' ' -f 1)" --date "$(date -I)" --specification-commit "$spec"

# provenance českého překladu auditovaného korpusu; workdir je pracovní adresář překladu, ve worktree absolutní cesta hlavního checkoutu
provenance-translation workdir="build/translation":
    {{runner}} mediparse-translation-provenance --workdir "{{workdir}}"

# build kontejnerového image přes podman, nebo docker
build:
    {{container-engine}} build -f Containerfile -t mediparse .
