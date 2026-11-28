#!/usr/bin/env bash
# Odešle úlohu PBS na MetaCentrum. Na charonu připraví ~/mediparse-<běh> s klonem repa na revizi REV
# (výchozí origin/main), skript úlohy nahraje jako job.sh a soubory z kořene hlavního checkoutu na stejné
# cesty v klonu, pak zavolá qsub a ID úlohy uloží do job.id. Úloha najde repo v $PBS_O_WORKDIR/repo
# a pixi v $PBS_O_WORKDIR/bin. Spouští jen uživatel ve svém terminálu, protože nahrává data MIMIC:
#   pixi run --frozen bash scripts/metacentrum-submit.sh <běh> <skript-úlohy> [cesta...]
#   REV=origin/s39t1-transformery-stupne-4 pixi run --frozen bash scripts/metacentrum-submit.sh \
#       stage4 job.sh resources/mimic/cohort.parquet resources/mimic/splits.parquet
# Hostitel charon je alias z ~/.ssh/config s ControlMaster, takže se přihlašuje jen jednou.
# Jméno běhu je ověřené regulárním výrazem, jeho rozvinutí na klientu v příkazech ssh je záměr.
# shellcheck disable=SC2029
set -euo pipefail

mediparse-data-access-check || {
    echo "Skript nahrává data MIMIC na MetaCentrum; pouští ho jen uživatel ve vlastním terminálu." >&2
    exit 1
}

run=${1:?chybí jméno běhu}
job=${2:?chybí skript úlohy}
shift 2
[[ $run =~ ^[a-z0-9-]+$ ]] || {
    echo "Jméno běhu smí mít jen malá písmena, číslice a pomlčku: $run" >&2
    exit 1
}
remote=mediparse-$run
private=--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=

echo "== 1/3 příprava ~/$remote na revizi ${REV:-origin/main}"
ssh charon "bash -s -- $(printf '%q ' "$remote" "${REV:-origin/main}")" <<'REMOTE'
set -euo pipefail
dir=~/$1
chmod 700 ~
umask 077
[[ -x ~/mediparse-translation/bin/pixi ]] || { echo "Chybí ~/mediparse-translation/bin/pixi (návod s21c, krok 1)" >&2; exit 1; }
mkdir -p "$dir/bin" "$dir/logs"
ln -sfn ~/mediparse-translation/bin/pixi "$dir/bin/pixi"
if [[ -d $dir/repo/.git ]]; then
    git -C "$dir/repo" fetch -q origin
else
    git clone -q https://github.com/Zohino/mediparse.git "$dir/repo"
fi
git -C "$dir/repo" checkout -q --detach "$2"
git -C "$dir/repo" log -1 --oneline
REMOTE

echo "== 2/3 nahrání skriptu úlohy a ${#} souborů"
rsync -a "$private" "$job" "charon:$remote/job.sh"
if (($#)); then
    rsync -aR --info=name "$private" "$@" "charon:$remote/repo/"
fi

echo "== 3/3 odeslání úlohy"
ssh charon "cd $remote && job=\$(qsub job.sh) && echo \"\$job\" | tee job.id && qstat -u \"\$USER\""

cat <<EOF

Po doběhnutí z kořene hlavního checkoutu:
  pixi run --frozen bash scripts/metacentrum-fetch.sh $run <cesta...>
EOF
