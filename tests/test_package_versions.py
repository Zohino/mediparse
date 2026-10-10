"""Zjištění verzí knihoven pro run manifest."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError

import pytest
import sklearn

from mediparse.infrastructure.package_versions import installed_versions


def test_reports_python_and_package_versions() -> None:
    """Python se čte z platform, ostatní z metadat distribuce."""
    versions = installed_versions(["python", "scikit-learn"])

    assert versions == {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
    }


def test_unknown_package_raises() -> None:
    """Neznámý balíček je programátorská chyba a projde výjimkou."""
    with pytest.raises(PackageNotFoundError):
        installed_versions(["no-such-package-xyz"])
