#!/usr/bin/env bash
set -euo pipefail

container_engine=${CONTAINER_ENGINE:-$(command -v podman || command -v docker || echo docker)}

dirty=false
[[ -z "$(git status --porcelain)" ]] || dirty=true
"$container_engine" build -f Containerfile --build-arg "MEDIPARSE_COMMIT=$(git rev-parse HEAD)" --build-arg "MEDIPARSE_DIRTY=$dirty" -t mediparse .
