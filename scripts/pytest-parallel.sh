#!/usr/bin/env bash
set -uo pipefail

work=$(mktemp -d) || exit 1
trap 'rm -rf "$work"' EXIT

pytest "$@" tests/test_download_script.py >"$work/download" 2>&1 &
download_pid=$!
pytest "$@" --ignore=tests/test_download_script.py >"$work/rest" 2>&1 &
rest_pid=$!

status=0
wait "$download_pid" || status=1
cat "$work/download"
wait "$rest_pid" || status=1
cat "$work/rest"
exit "$status"
