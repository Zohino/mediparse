"""Návratové kódy vstupních bodů, společné pro všechny nástroje mediparse."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Návratový kód procesu: kontrola prošla, zablokovala, nebo odmítla běžet."""

    OK = 0
    BLOCKED = 1
    REFUSED = 2
