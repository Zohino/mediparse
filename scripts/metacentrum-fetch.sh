#!/usr/bin/env bash
# Stáhne výsledky úlohy z ~/mediparse-<běh>/repo na charonu na stejné cesty v kořeni hlavního checkoutu
# a nabídne smazání resources/mimic na charonu. Dokud úloha čeká nebo běží, nic nestahuje.
# Spouští jen uživatel ve svém terminálu, protože stahuje i data MIMIC:
#   pixi run --frozen bash scripts/metacentrum-fetch.sh <běh> <cesta...>
#   pixi run --frozen bash scripts/metacentrum-fetch.sh stage4 results/screening resources/mimic/screening
# Hostitel charon je alias z ~/.ssh/config s ControlMaster, takže se přihlašuje jen jednou.
# Jméno běhu je ověřené regulárním výrazem, jeho rozvinutí na klientu v příkazech ssh je záměr.
# shellcheck disable=SC2029
set -euo pipefail

mediparse-data-access-check || {
    echo "Skript stahuje data MIMIC z MetaCentra; pouští ho jen uživatel ve vlastním terminálu." >&2
    exit 1
}

run=${1:?chybí jméno běhu}
shift
(($#)) || {
    echo "Chybí cesty ke stažení" >&2
    exit 1
}
[[ $run =~ ^[a-z0-9-]+$ ]] || {
    echo "Jméno běhu smí mít jen malá písmena, číslice a pomlčku: $run" >&2
    exit 1
}
remote=mediparse-$run

job=$(ssh charon "cat $remote/job.id")
# Bez -x zná qstat jen čekající a běžící úlohy; doběhlé ukáže -x nejvýš 24 h.
if ssh charon "qstat $job" >/dev/null 2>&1; then
    echo "Úloha $job ještě čeká nebo běží; nic nestahuji." >&2
    exit 1
fi
ssh charon "qstat -xf $job 2>/dev/null | /usr/bin/grep -E 'job_state|Exit_status|walltime' || true"

# /./ říká rsyncu -R, od kterého místa se cesta zachová i lokálně.
sources=()
for path in "$@"; do
    sources+=("charon:$remote/repo/./$path")
done
umask 077
rsync -aR --info=name "${sources[@]}" .

read -rp "Smazat ~/$remote/repo/resources/mimic na charonu? [a/N] " answer
if [[ $answer == a ]]; then
    ssh charon "rm -rv $remote/repo/resources/mimic"
fi
