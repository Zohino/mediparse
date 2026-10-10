#!/usr/bin/env bash
set -euo pipefail

dir=${1:?chybí adresář s tabulkami MIMIC}

jq -r --arg dir "$dir" '.mimic_tables[] | "\(.sha256)  \($dir)/\(.url | split("/") | last)"' config/mimic_tables.json | sha256sum --check --strict
