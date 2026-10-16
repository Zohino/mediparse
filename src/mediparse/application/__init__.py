"""Use casy mediparse: co se v jednom kroku děje, bez vazby na technologie; potřeby popisují porty.

Use case smí volat funkci jiného use casu (S11s), port ale ne: port, který používá
víc use casů, leží v balíčku ``ports`` podle kontextu. Kroky pipeline za sebou řadí
Snakemake, ne use case.

Signalizace odmítnutí, kterou dodržuje každý use case:

- chybný vstup, config nebo data jsou ``InvalidInputError`` propuštěná ke vstupnímu
  bodu, který ji převede na REFUSED; chyby čtení z adaptérů use case nechytá ani
  nevrací jako hodnotu;
- věcný výsledek kontroly je outcome typ, který vstupní bod mapuje na OK, BLOCKED
  nebo REFUSED; outcome s odmítnutím má jen use case, jehož výsledkem je výčet stavů
  kontroly (audit, provenance), a jen pro předpoklady, které ověřuje sám;
- porušený invariant programu je obyčejná výjimka s tracebackem;
- ``run`` nevrací ``None``;
- data configu jsou argumenty ``run``, v polích dataclass jsou jen porty.
"""
