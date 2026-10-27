"""Styl a vektorový export grafů pro sazbu práce."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.rcsetup import cycler
from matplotlib.ticker import FuncFormatter

from mediparse.domain.czech import NBSP, number_phrase

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap

THESIS_FONT: Final = "Libertinus Serif"
PALETTE_CATEGORICAL: Final = ("#0072B2", "#E69F00", "#000000", "#F0E442", "#D55E00")
THESIS_STYLE: Final[Mapping[str, object]] = {
    "font.family": THESIS_FONT,
    "font.size": 10,
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
def figure(
    height_cm: float, *, font: str = THESIS_FONT
) -> Generator[tuple[Figure, Axes]]:
    """Vytvoří figuru širokou 16,5 cm a osy v dočasném stylu práce.

    Popisky i kresbu přidávejte uvnitř kontextu. Legendu vně vpravo vytvoří
    ``figure.legend(loc="outside right upper")``; metoda os outside neumí.
    Figura nevstupuje do globálního registru pyplotu.

    Args:
        height_cm: Výška figury v centimetrech.
        font: Rodina písma; výchozí odpovídá textu práce.

    Yields:
        Figura a její osy připravené ke kreslení.
    """
    with mpl.rc_context(thesis_style(font=font)):
        plot = Figure(figsize=(16.5 / 2.54, height_cm / 2.54))
        yield plot, plot.subplots()


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
    colors = mpl.colormaps["viridis"](np.linspace(0.45, 1, 256))[::-1]
    return ListedColormap(colors).with_extremes(bad="white")
