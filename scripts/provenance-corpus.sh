#!/usr/bin/env bash
set -euo pipefail

model=${1:?chybí název modelu}

[[ -z "$(git status --porcelain -- docs/synthetic-corpus.md)" ]] || {
    echo "Specifikace má necommitnuté změny." >&2
    exit 1
}
spec=$(git log -1 --format=%H -- docs/synthetic-corpus.md)
git merge-base --is-ancestor "$spec" origin/main || {
    echo "Commit specifikace $spec není v origin/main; rebase merge by ho přepsal, změnu specifikace merguj dřív." >&2
    exit 1
}
mediparse-corpus-provenance --model "$model" --claude-code-version "$(claude --version | cut -d ' ' -f 1)" --date "$(date -I)" --specification-commit "$spec"
