#!/usr/bin/env bash
set -euo pipefail

version=7.051
sha256=4d9be29b5cb380c35af8ba967abcc752ad1e07be1f738a9789c33e0dd7478c92
target=${XDG_DATA_HOME:-$HOME/.local/share}/fonts/libertinus
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

curl --fail --silent --show-error --location \
    --output "$work/libertinus.zip" \
    "https://github.com/alerque/libertinus/releases/download/v$version/Libertinus-$version.zip"
echo "$sha256  $work/libertinus.zip" | sha256sum --check --quiet
unzip -q -j "$work/libertinus.zip" "Libertinus-$version/static/OTF/LibertinusSerif-*.otf" -d "$work/otf"
mkdir -p "$target"
cp "$work"/otf/*.otf "$target"/
echo "Libertinus Serif $version: $target"
