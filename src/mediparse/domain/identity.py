"""Identita řádku matice, společná kontextům, které ho nesou: trénink, evaluace a manifest."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

RowId = Annotated[str, Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]
