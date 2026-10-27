"""Styl a vektorový export grafů pro sazbu práce."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Final

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.rcsetup import cycler
from matplotlib.ticker import FuncFormatter

from mediparse.domain.czech import NBSP, number_phrase
from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path

    from matplotlib.colors import Colormap

THESIS_WIDTH_CM: Final = 16.5
THESIS_FONT: Final = "Libertinus Serif"
THESIS_FONT_SIZE: Final = 10
SEQUENTIAL_BEGIN: Final = 0.45
FONT_INSTALLATION: Final = (
    "Nainstalujte Libertinus Serif: Fedora libertinus-fonts, Debian fonts-libertinus, "
    "jinde bash scripts/install-libertinus.sh."
)
# Matplotlib zapisuje jména písem nekomprimovaně, prefix ABCDEF+ označuje podmnožinu.
BASE_FONT: Final = re.compile(rb"/BaseFont /(?:[A-Z]{6}\+)?([A-Za-z0-9-]+)")
PALETTE_CATEGORICAL: Final = ("#0072B2", "#E69F00", "#000000", "#F0E442", "#D55E00")
THESIS_STYLE: Final[Mapping[str, object]] = {
    "font.family": THESIS_FONT,
    "font.size": THESIS_FONT_SIZE,
    "axes.prop_cycle": cycler(color=PALETTE_CATEGORICAL),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#DDDDDD",
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "savefig.pad_inches": 0,
    # OTF CFF s fonttype 42 vkládá nesoulad typu písma a jeho souboru do PDF.
    "pdf.fonttype": 3,
    "axes.formatter.use_locale": False,
}


def thesis_style(*, font: str = THESIS_FONT) -> mpl.RcParams:
    """Vrátí rcParams práce s vybranou rodinou písma.

    Returns:
        Samostatné nastavení vhodné pro rc_context.
    """
    return mpl.RcParams(**{**THESIS_STYLE, "font.family": font})


@contextmanager
def figure(  # ruff: ignore[too-many-arguments] - explicitní volby rozvržení podle Figure.subplots
    height_cm: float,
    *,
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    font: str = THESIS_FONT,
) -> Generator[tuple[Figure, Any]]:
    """Vytvoří figuru širokou 16,5 cm a osy v dočasném stylu práce.

    Popisky i kresbu přidávejte uvnitř kontextu. Legendu vně vpravo vytvoří
    ``figure.legend(loc="outside right upper")``; metoda os outside neumí.
    Figura nevstupuje do globálního registru pyplotu.

    Args:
        height_cm: Výška figury v centimetrech.
        nrows: Počet řádků panelů.
        ncols: Počet sloupců panelů.
        sharex: Zda panely sdílejí osu x.
        sharey: Zda panely sdílejí osu y.
        font: Rodina písma; výchozí odpovídá textu práce.

    Yields:
        Figura a její osy připravené ke kreslení.

    Raises:
        InvalidInputError: Požadované písmo není dostupné.
    """
    register_thesis_font(font)
    with mpl.rc_context(thesis_style(font=font)):
        plot = Figure(figsize=(THESIS_WIDTH_CM / 2.54, height_cm / 2.54))
        # subplots vrací podle rozměrů Axes nebo pole různých tvarů; Any zachovává toto rozhraní.
        yield (
            plot,
            plot.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey),
        )


def number_axis() -> FuncFormatter:
    """Vytvoří formátovač osy s českým zápisem čísel.

    Returns:
        Formátovač s desetinnou čárkou a NBSP mezi tisíci.
    """
    return FuncFormatter(lambda value, _position: number_phrase(value))


def percent_axis() -> FuncFormatter:
    """Vytvoří formátovač podílů jako českých procent.

    Returns:
        Formátovač s NBSP před znakem procenta.
    """
    return FuncFormatter(
        lambda value, _position: f"{number_phrase(100 * value)}{NBSP}%"
    )


def sequential_colormap() -> Colormap:
    """Vytvoří obrácenou zkrácenou viridis s bílou pro chybějící hodnoty.

    Returns:
        Škála umožňující černé popisky i na nejtmavší buňce.
    """
    colors = mpl.colormaps["viridis"](np.linspace(SEQUENTIAL_BEGIN, 1, 256))[::-1]
    return ListedColormap(colors).with_extremes(bad="white")


def _find_font(font: str) -> str:
    try:
        return font_manager.fontManager.findfont(
            font_manager.FontProperties(family=font), fallback_to_default=False
        )
    except ValueError as error:
        msg = f"Písmo {font!r} není dostupné. {FONT_INSTALLATION}"
        raise InvalidInputError(msg) from error


def register_thesis_font(font: str = THESIS_FONT) -> None:
    """Zaregistruje systémové řezy zadané rodiny i při zastaralé cache.

    Args:
        font: Rodina, jejíž řezy se mají zaregistrovat.

    Raises:
        InvalidInputError: Písmo práce není dostupné.
    """
    registered = {entry.fname for entry in font_manager.fontManager.ttflist}
    for path in font_manager.findSystemFonts():
        if path in registered:
            continue
        if font_manager.get_font(path).family_name == font:
            # addfont připojuje záznamy i opakovaně; kontrola cesty brání duplikátům.
            font_manager.fontManager.addfont(path)
    _find_font(font)


def embedded_fonts(pdf: bytes) -> frozenset[str]:
    """Přečte jména všech písem z PDF vytvořeného matplotlibem.

    Returns:
        PostScriptová jména řezů bez prefixů podmnožin.
    """
    return frozenset(name.decode("ascii") for name in BASE_FONT.findall(pdf))


def save_pdf(figure: Figure, path: Path, *, font: str = THESIS_FONT) -> None:
    """Uloží vektorové PDF a odmítne cizí vložená písma.

    Args:
        figure: Hotový graf vytvořený ve stylu práce.
        path: Cílový soubor v existujícím adresáři.
        font: Jediná povolená rodina písma.

    Raises:
        InvalidInputError: Písmo chybí nebo PDF obsahuje jinou rodinu;
            neplatné PDF se po kontrole smaže.
    """
    register_thesis_font(font)
    font_path = _find_font(font)
    family = font_manager.get_font(font_path).postscript_name.split("-")[0]
    with mpl.rc_context(thesis_style(font=font)):
        figure.savefig(path, format="pdf")
    fonts = embedded_fonts(path.read_bytes())
    foreign = {
        name for name in fonts if name != family and not name.startswith(f"{family}-")
    }
    if not foreign:
        return
    path.unlink()
    found = ", ".join(sorted(foreign))
    msg = f"PDF neobsahuje pouze rodinu {font!r}: {found}. {FONT_INSTALLATION}"
    raise InvalidInputError(msg)
