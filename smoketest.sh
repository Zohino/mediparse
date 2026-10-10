#!/bin/sh
set -eu

cd "$(dirname "$0")"

fail() {
    echo "smoketest.sh: $1" >&2
    exit 1
}

command -v git >/dev/null 2>&1 || fail "git chybí; nainstaluj git a repozitář naklonuj přes git clone."
git_error=$(git rev-parse --git-dir 2>&1) || fail "git adresář nepovažuje za repozitář ($git_error); naklonuj repo přes git clone, archiv ZIP nestačí, protože run manifest potřebuje commit."

engine=${CONTAINER_ENGINE:-}
if [ -z "$engine" ]; then
    if command -v podman >/dev/null 2>&1; then
        engine=podman
    elif command -v docker >/dev/null 2>&1; then
        engine=docker
    else
        fail "chybí podman i docker; nainstaluj jeden z nich nebo nastav CONTAINER_ENGINE."
    fi
fi

commit=$(git rev-parse HEAD)
status=$(git status --porcelain) || fail "git status selhal."
dirty=false
if [ -n "$status" ]; then
    dirty=true
fi

set -- run --rm --network none
case "$(basename "$engine")" in
    docker*)
        if [ "$(uname -s)" = Linux ]; then
            set -- "$@" --user "$(id -u):$(id -g)"
        fi
        ;;
esac

export MSYS_NO_PATHCONV=1
root=$(pwd -W 2>/dev/null || pwd)

"$engine" build -f Containerfile \
    --build-arg "MEDIPARSE_COMMIT=$commit" \
    --build-arg "MEDIPARSE_DIRTY=$dirty" \
    -t mediparse .

mkdir -p build logs

set -- "$@" -v "$root/build:/app/build" -v "$root/logs:/app/logs" mediparse
"$engine" "$@"
