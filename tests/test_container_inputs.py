"""Soubory nezbytné pro testy spouštěné v obrazu kontejneru."""

import pytest

from tests.support import REPOSITORY

BUILD_CONTEXT_PRESENT = pytest.mark.skipif(
    not (REPOSITORY / "Containerfile").is_file(),
    reason="Kontext sestavení ověřuje hostitelské CI; v běhovém obrazu není.",
)


def _allowed_paths(dockerignore: str) -> set[str]:
    return {line[1:] for line in dockerignore.splitlines() if line.startswith("!")}


def _copied_paths(containerfile: str) -> set[str]:
    copied: set[str] = set()
    for line in containerfile.splitlines():
        tokens = line.split()
        if not tokens or tokens[0] != "COPY":
            continue
        if line.rstrip().endswith("\\"):
            message = f"Víceřádkové COPY není podporováno: {line}"
            raise ValueError(message)
        if any(token.startswith("--from=") for token in tokens[1:-1]):
            continue
        copied.update(tokens[1:-1])
    return copied


def _mismatch(dockerignore: str, containerfile: str) -> dict[str, list[str]]:
    allowed = _allowed_paths(dockerignore)
    copied = _copied_paths(containerfile)
    return {
        "povoleno, nekopíruje se": sorted(allowed - copied),
        "kopíruje se, nepovoleno": sorted(copied - allowed),
    }


@pytest.mark.parametrize(
    ("dockerignore", "containerfile", "missing_copy", "missing_allow"),
    [
        ("!config/split.json\n!src/\n", "COPY src/ src/\n", ["config/split.json"], []),
        (
            "!config/split.json\n!config/novy.json\n",
            "COPY config/split.json config/\n",
            ["config/novy.json"],
            [],
        ),
        (
            "!src/\n",
            "COPY config/split.json config/\nCOPY src/ src/\n",
            [],
            ["config/split.json"],
        ),
        (
            "!config/novy.json\n",
            "COPY config/split.json config/\n",
            ["config/novy.json"],
            ["config/split.json"],
        ),
    ],
    ids=["missing-copy", "new-allowlist-entry", "missing-allowlist-entry", "both"],
)
def test_container_input_mismatch_names_paths(
    dockerignore: str,
    containerfile: str,
    missing_copy: list[str],
    missing_allow: list[str],
) -> None:
    """Neshoda pojmenuje chybějící cesty a směr rozdílu."""
    assert _mismatch(dockerignore, containerfile) == {
        "povoleno, nekopíruje se": missing_copy,
        "kopíruje se, nepovoleno": missing_allow,
    }


def test_dockerignore_includes_only_allowlisted_paths() -> None:
    """Zákazy, komentáře a prázdné řádky nejsou součástí allowlistu."""
    source = "*\n\n# Kontext sestavení\n!src/\n!config/split.json\n**/__pycache__/\n"
    assert _allowed_paths(source) == {"src/", "config/split.json"}


def test_container_copy_sources_exclude_stage_copies() -> None:
    """Zdrojové cesty zahrnují všechny fáze, ale vynechají COPY z jiné fáze."""
    source = (
        "FROM image AS builder\n\n# Zdroje z kontextu\n"
        "COPY pyproject.toml pixi.lock ./\n"
        "COPY src/ src/\n"
        "FROM image AS runtime\n"
        "COPY --from=builder /app/src/ src/\n"
        "COPY config/split.json config/\n"
    )
    assert _copied_paths(source) == {
        "pyproject.toml",
        "pixi.lock",
        "src/",
        "config/split.json",
    }


@pytest.mark.parametrize(
    "source",
    [
        "COPY config/split.json \\\n    config/\n",
        "COPY config/split.json \\  \n    config/\n",
    ],
    ids=["backslash", "backslash-trailing-spaces"],
)
def test_container_copy_rejects_line_continuation(source: str) -> None:
    """Víceřádkové COPY nesmí potichu změnit parsovanou množinu.

    Docker i podman berou za pokračování i zpětné lomítko s mezerami za ním.
    """
    with pytest.raises(ValueError, match="Víceřádkové COPY není podporováno"):
        _copied_paths(source)


@BUILD_CONTEXT_PRESENT
def test_container_allowlist_matches_copy_sources() -> None:
    """Každá povolená cesta se kopíruje do obrazu a každá kopírovaná je povolená."""
    assert _mismatch(
        (REPOSITORY / ".dockerignore").read_text(encoding="utf-8"),
        (REPOSITORY / "Containerfile").read_text(encoding="utf-8"),
    ) == {"povoleno, nekopíruje se": [], "kopíruje se, nepovoleno": []}


@BUILD_CONTEXT_PRESENT
def test_translation_prepare_is_available_to_container_tests() -> None:
    """Příprava překladu projde kontextem sestavení i COPY do běhového obrazu."""
    required = "translation/prepare.py"
    included = _allowed_paths(
        (REPOSITORY / ".dockerignore").read_text(encoding="utf-8")
    )
    copied = _copied_paths((REPOSITORY / "Containerfile").read_text(encoding="utf-8"))
    assert required in included
    assert required in copied
