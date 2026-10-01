#!/usr/bin/env bash
set -euo pipefail

fail() {
    echo "$1" >&2
    exit "$2"
}

(($# == 4)) || fail "použití: $0 NETRC URL VÝSTUP LOG" 2

netrc=$1
url=$2
output=$3
log=$4
partial=$output.part
attempts=$log.part

[[ -r $netrc ]] || fail "chybí $netrc s přihlašovacími údaji" 1
[[ -z $(find -L "$netrc" -perm /077) ]] || fail "$netrc smí číst jen vlastník: chmod 600 $netrc" 1

echo "== $(date --utc +%FT%TZ) stahuji $url" >>"$attempts"

wget2 \
    --no-config \
    --netrc-file="$netrc" \
    --continue \
    --progress=none \
    --output-document="$partial" \
    "$url" \
    >>"$attempts" 2>&1 || {
    status=$?
    case $status in
        6) fail "server odmítl přihlášení k $url, zkontroluj $netrc (detail v $attempts)" "$status" ;;
        8) fail "server odmítl $url, např. 403 bez přístupu nebo 404 špatná URL (detail v $attempts)" "$status" ;;
        *) fail "stažení $url selhalo s kódem $status (detail v $attempts)" "$status" ;;
    esac
}

mv -- "$partial" "$output"
echo "== $(date --utc +%FT%TZ) staženo do $output" >>"$attempts"
cat -- "$attempts" >>"$log"
rm -- "$attempts"
