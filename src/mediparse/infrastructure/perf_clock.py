"""Monotonní hodiny s nejvyšším dostupným rozlišením pro krátké intervaly."""

from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class PerfCounterClock:
    """Adaptér hodin nad čítačem výkonu systému."""

    @staticmethod
    def now() -> float:
        """Přečte monotonní čas v sekundách.

        Returns:
            Stav čítače, jehož rozdíl mezi voláními je uplynulý čas.
        """
        return perf_counter()
