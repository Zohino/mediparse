#!/usr/bin/env bash
set -uo pipefail
# Bez řízení úloh asynchronní příkaz ignoruje SIGINT; pytest to předá testům,
# které SIGINT posílají svým podprocesům.
set -m

# Testy stahování čekají na reálné časové limity aria2c, proto běží vedle
# výpočetních; jejich výstup vypíšeme až nakonec, aby se bloky nepromíchaly.
slow=tests/test_download_script.py
output=$(mktemp) || exit 1

pytest "$@" "$slow" >"$output" 2>&1 &
slow_pid=$!
trap 'kill -- "-$slow_pid" 2>/dev/null || :; wait "$slow_pid" 2>/dev/null || :; cat "$output"; rm -f "$output"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

status=0
pytest "$@" --ignore="$slow" 2>&1 || status=1
wait "$slow_pid" || status=1
exit "$status"
