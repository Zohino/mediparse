"""Identita pacienta a řádku matice, společná kontextům, které je nesou: kohorta, audit, evaluace a manifest."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

SubjectId = Annotated[int, Field(gt=0)]
RowId = Annotated[str, Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]
