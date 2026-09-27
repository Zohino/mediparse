"""Smoke test: balíček mediparse jde naimportovat z nainstalovaného prostředí."""

import importlib


def test_mediparse_imports() -> None:
    """Balíček se naimportuje a nese modulový docstring."""
    module = importlib.import_module("mediparse")

    assert module.__doc__
