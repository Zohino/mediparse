"""Identita pacienta a řádku matice jako omezená primitiva domény."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

# Omezení validuje jen pydantic model; na dataclassech je anotace dokumentací pojmu.
SubjectId = Annotated[int, Field(gt=0)]
RowId = Annotated[str, Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]
