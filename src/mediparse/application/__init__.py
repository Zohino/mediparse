"""Use casy mediparse: co se v jednom kroku děje, bez vazby na technologie; potřeby popisují porty.

Use case smí volat funkci jiného use casu (S11s), port ale ne: port, který používá
víc use casů, leží v balíčku ``ports`` podle kontextu. Kroky pipeline za sebou řadí
Snakemake, ne use case.
"""
