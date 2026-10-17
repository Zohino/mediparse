"""Seed generátoru náhodných čísel."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

# Rozsah random_state v numpy a scikit-learn; seed mimo něj by skončil
# tracebackem v adaptéru místo odmítnutí configu.
Seed = Annotated[int, Field(ge=0, lt=2**32)]
