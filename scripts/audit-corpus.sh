#!/usr/bin/env bash
set -euo pipefail

discharge=${1:?chybí cesta k referenčním zprávám discharge}
radiology=${2:?chybí cesta k referenčním zprávám radiology}
report=${3:?chybí cesta k záznamu auditu}

root=${PIXI_PROJECT_ROOT:-$(git rev-parse --show-toplevel)}
python -I -c '
import sys
from pathlib import Path
from mediparse.infrastructure.source_revision import GitRevision
sys.exit(GitRevision(Path(sys.argv[1]), excluded=("resources/synthetic",)).read().dirty)
' "$root" || {
    echo "Repo má necommitnutý kód mimo korpus; záznam auditu by ukazoval na jiný kód." >&2
    exit 1
}
mediparse-corpus-audit --reference "$discharge" --reference "$radiology" --report "$report" --commit "$(git rev-parse HEAD)"
