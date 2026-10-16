"""Otisky: celý SHA-1 commitu a otisk sady souborů."""

import pytest

from mediparse.domain.digest import commit_sha, fingerprint


def test_fingerprint_ignores_order_but_not_content_or_names() -> None:
    """Otisk nezávisí na pořadí souborů, ale na jejich obsahu i jménech ano."""
    files = [("en/a.txt", b"one"), ("cs/a.txt", b"two")]
    base = fingerprint(files)

    assert fingerprint(reversed(files)) == base
    assert fingerprint([("en/a.txt", b"one!"), ("cs/a.txt", b"two")]) != base
    assert fingerprint([("en/b.txt", b"one"), ("cs/a.txt", b"two")]) != base


@pytest.mark.parametrize("value", ["abc123", "HEAD", "C" * 40, "c" * 41])
def test_commit_sha_rejects_anything_but_full_hash(value: str) -> None:
    """Commit je jen celý SHA-1 hash malými písmeny."""
    with pytest.raises(ValueError, match="SHA-1"):
        commit_sha(value)
