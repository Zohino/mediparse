#!/usr/bin/env bash
set -euo pipefail

if (($# != 2)); then
    echo "Použití: check-pr-record.sh <titulek PR> <soubor s tělem PR>" >&2
    exit 2
fi

title=$1
body=$2
types='feat|fix|refactor|test|build|docs|ci|style|perf'
failed=0

if [[ ! $title =~ ^($types)(\([a-z0-9_/-]+\))?:\ .+\ \(S[0-9]+[a-z0-9]*\)$ ]]; then
    echo "Titulek PR nemá tvar 'typ(scope): předmět (SNN)', typ je jeden z $types: $title" >&2
    failed=1
fi

if ! tr -d '\r' <"$body" | grep -Eq '^Task: S[0-9]+$'; then
    echo "Tělo PR nemá řádek 'Task: SNN', bez něj /start nenajde commit tasku na main." >&2
    failed=1
fi

exit "$failed"
