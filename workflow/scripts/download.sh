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
retry_wait=${MEDIPARSE_DOWNLOAD_RETRY_WAIT:-10}
period=${MEDIPARSE_DOWNLOAD_PERIOD:-1200}
tries=5

# Mrtvý server musí vyčerpat pokusy všech spojení dřív, než aria2c skončí
# plánovaně, jinak by se neúspěšný pokus nikdy nezapočítal.
((period > tries * (timeout + retry_wait))) \
    || fail "MEDIPARSE_DOWNLOAD_PERIOD ($period s) musí být delší než $tries × (timeout + čekání)" 2

[[ -r $netrc ]] || fail "chybí $netrc s přihlašovacími údaji" 1
[[ -z $(find -L "$netrc" -perm /077) ]] || fail "$netrc smí číst jen vlastník: chmod 600 $netrc" 1

# PhysioNet škrtí rychlost na jedno spojení, proto aria2c stahuje osmi spojeními
# po kusech. Zaseknuté spojení ukončí --timeout a aria2c ho otevře znovu
# (--max-tries). --lowest-speed-limit chybí záměrně: zaseknutá spojení ukončí
# dřív než timeout a už je neobnoví, takže souběh klesne až na nulu (běh
# 2026-10-23). Pokusy ale aria2c počítá za celý život spojení, ne po sobě
# jdoucí chyby, takže na mnohahodinovém stažení spojení postupně vypadávají
# taky. Proto aria2c po $period s skončí (--stop) a smyčka níže ho spustí znovu
# s čerstvými spojeními. Když vyčerpají pokusy všechna spojení, smyčka naváže
# podle kontrolního souboru $partial.aria2. Průběh jde na terminál (stdout),
# do logu jen hlášení. PhysioNet nabídne přihlášení z .netrc jen User-Agentu
# začínajícímu „Wget/“ (ostatním vrátí 403 i s platným heslem), proto se aria2c
# hlásí jako Wget a v závorce uvádí, kdo skutečně stahuje.
download() {
    started=$SECONDS
    aria2c \
        --no-conf \
        --netrc-path="$netrc" \
        --user-agent="Wget/1.21 (aria2)" \
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
        --max-tries="$tries" \
        --retry-wait="$retry_wait" \
        --stop="$period" \
        --show-console-readout=false \
        --summary-interval=60 \
        --console-log-level=warn \
        --log="$attempts" \
        --log-level=notice \
        "$url" \
        2>&1
}

# Kódy aria2c (manuál, EXIT STATUS): 2 timeout, 6 síť,
# 19 DNS, 29 přetížený server. Spojení zavřené před koncem těla („Got EOF
# from the server“) hlásí aria2c jen obecným kódem 1. Ostatní chyby se
# opakováním nespraví.
transient() {
    case $1 in
        1 | 2 | 6 | 19 | 29) return 0 ;;
        *) return 1 ;;
    esac
}

# aria2c po Ctrl+C vrátí kód 2, pokud už předtím vypadlo spojení timeoutem,
# a smyčka by ho brala jako přechodnou chybu. Bash spustí trap až po skončení
# aria2c, takže kontrolní soubor zůstane uložený.
trap 'fail "stažení $url přerušeno; další běh naváže" 130' INT TERM

echo "== $(date --utc +%FT%TZ) stahuji $url" >>"$attempts"

attempt=1
until download; do
    status=$?
    # Plánované skončení po --stop vrací aria2c stejné kódy jako chyby (7, po
    # vypadlém spojení 2), pozná se proto jen podle uplynulého času.
    if ((SECONDS - started >= period)); then
        echo "aria2c stahuje $period s, spouštím ho znovu s čerstvými spojeními" >&2
        continue
    fi
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
