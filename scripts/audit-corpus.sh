#!/usr/bin/env bash
set -euo pipefail

discharge=${1:?chybí cesta k referenčním zprávám discharge}
radiology=${2:?chybí cesta k referenčním zprávám radiology}
report=${3:?chybí cesta k záznamu auditu}

[[ -z "$(git status --porcelain -- . ':!resources/synthetic')" ]] || {
    echo "Repo má mimo korpus necommitnuté změny; záznam auditu by ukazoval na jiný kód." >&2
    exit 1
}
mediparse-corpus-audit --reference "$discharge" --reference "$radiology" --report "$report" --commit "$(git rev-parse HEAD)"
