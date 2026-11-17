#!/usr/bin/env bash
set -uo pipefail

# Čekací testy s reálnými limity (~112 s ze ~230 s) běží vedle výpočetních;
# jejich výstup vypíšeme až nakonec, aby se bloky nepromíchaly.
slow=tests/test_download_script.py
output=$(mktemp) || exit 1

pytest "$@" "$slow" >"$output" 2>&1 &
slow_pid=$!
trap 'kill "$slow_pid" 2>/dev/null || :; wait "$slow_pid" 2>/dev/null || :; rm -f "$output"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

status=0
pytest "$@" --ignore="$slow" 2>&1 || status=1
wait "$slow_pid" || status=1
cat "$output"
exit "$status"
