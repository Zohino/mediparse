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

# Výchozí hodnoty míří na PhysioNet; testy je zkracují proměnnými prostředí.
timeout=${MEDIPARSE_DOWNLOAD_TIMEOUT:-60}
max_attempts=${MEDIPARSE_DOWNLOAD_ATTEMPTS:-50}
retry_wait=${MEDIPARSE_DOWNLOAD_RETRY_WAIT:-60}

[[ -r $netrc ]] || fail "chybí $netrc s přihlašovacími údaji" 1
[[ -z $(find -L "$netrc" -perm /077) ]] || fail "$netrc smí číst jen vlastník: chmod 600 $netrc" 1

# PhysioNet škrtí rychlost na jedno spojení, proto aria2c stahuje osmi spojeními
# po kusech. Zaseknuté spojení ukončí --timeout, a když stojí všechna,
# --lowest-speed-limit ukončí celé stažení kódem 5; smyčka pak naváže podle
# kontrolního souboru $partial.aria2. Průběh jde na terminál (stdout),
# do logu jen hlášení.
download() {
    aria2c \
        --no-conf \
        --netrc-path="$netrc" \
        --dir="$(dirname -- "$partial")" \
        --out="$(basename -- "$partial")" \
        --continue=true \
        --always-resume=true \
        --allow-overwrite=false \
        --auto-file-renaming=false \
        --max-connection-per-server=8 \
        --split=8 \
        --min-split-size=1M \
        --timeout="$timeout" \
        --connect-timeout="$timeout" \
        --lowest-speed-limit=1K \
        --max-tries=5 \
        --retry-wait=10 \
        --show-console-readout=false \
        --summary-interval=60 \
        --console-log-level=warn \
        --log="$attempts" \
        --log-level=notice \
        "$url" \
        2>&1
}

# Kódy aria2c (manuál, EXIT STATUS): 2 timeout, 5 příliš pomalé, 6 síť,
# 19 DNS, 29 přetížený server. Spojení zavřené před koncem těla („Got EOF
# from the server“) hlásí aria2c jen obecným kódem 1. Ostatní chyby se
# opakováním nespraví.
transient() {
    case $1 in
        1 | 2 | 5 | 6 | 19 | 29) return 0 ;;
        *) return 1 ;;
    esac
}

echo "== $(date --utc +%FT%TZ) stahuji $url" >>"$attempts"

attempt=1
until download; do
    status=$?
    case $status in
        24) fail "server odmítl přihlášení k $url, zkontroluj $netrc (detail v $attempts)" "$status" ;;
        3 | 22) fail "server odmítl $url, např. 403 bez přístupu nebo 404 špatná URL (detail v $attempts)" "$status" ;;
    esac
    if ! transient "$status" || ((attempt >= max_attempts)); then
        fail "stažení $url selhalo s kódem $status po $attempt pokusech; další běh naváže (detail v $attempts)" "$status"
    fi
    echo "pokus $attempt/$max_attempts o $url skončil kódem $status, navazuji za $retry_wait s" >&2
    sleep "$retry_wait"
    attempt=$((attempt + 1))
done

# aria2c po úspěchu kontrolní soubor maže; když zůstal, soubor není celý.
[[ ! -e $partial.aria2 ]] || fail "stažení $url skončilo bez chyby, ale $partial.aria2 zůstal (detail v $attempts)" 1

mv -- "$partial" "$output"
echo "== $(date --utc +%FT%TZ) staženo do $output" >>"$attempts"
cat -- "$attempts" >>"$log"
rm -- "$attempts"
