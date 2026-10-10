FROM ghcr.io/prefix-dev/pixi:0.73.0@sha256:702f3a3c966aa63a0a57906466ec33ce9a463b532446f5584498fa429d069fd8 AS builder

WORKDIR /app

COPY pyproject.toml pixi.lock README.md LICENSE ./
RUN pixi install --locked -e test --skip mediparse

COPY src/ src/

RUN pixi install --locked -e test \
    && pixi shell-hook -e test -s bash > /shell-hook \
    && echo 'exec "$@"' >> /shell-hook

FROM docker.io/library/debian:trixie-slim@sha256:918311b7b6c4c6f68b232ba516584925f6c78ad82b6fd534b98979df6438e483 AS runtime

RUN apt-get update \
    && apt-get install --no-install-recommends -y wget2=2.2.0+ds-1+deb13u1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.pixi/envs/test /app/.pixi/envs/test
COPY --from=builder /shell-hook /shell-hook
COPY --from=builder /app/src/ src/
COPY pyproject.toml ./
COPY config/matrix.json config/smoketest_training.json config/synthetic_plan.json config/mimic_tables.json config/verbalization_template.md config/
COPY resources/synthetic/ resources/synthetic/
COPY translation/note_parts.py translation/markers.py translation/collect.py translation/
COPY workflow/ workflow/
COPY tests/ tests/

ARG MEDIPARSE_COMMIT=
ARG MEDIPARSE_DIRTY=

RUN mkdir -m 1777 .snakemake build logs

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    USER=mediparse

ENV MEDIPARSE_COMMIT=$MEDIPARSE_COMMIT \
    MEDIPARSE_DIRTY=$MEDIPARSE_DIRTY

ENTRYPOINT ["/bin/bash", "/shell-hook"]
CMD ["snakemake", "--forceall", "smoketest"]
