"""Chování stylu grafů, českých os a barevných škál práce."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import thesis_figure

if TYPE_CHECKING:
    from pathlib import Path

TEST_FONT = "DejaVu Serif"
FONT_SIZE = 10
PDF_FONT_TYPE = 3
PALETTE = ("#0072B2", "#E69F00", "#000000", "#F0E442", "#D55E00")


def test_figure_dimensions_style_and_legend() -> None:
    """Graf má sazební rozměr, paletu, mřížku a legendu vně vpravo."""
    with thesis_figure.figure(6, font=TEST_FONT) as (figure, axes):
        lines = [axes.plot([0, 1], [0, i], label=str(i))[0] for i in range(5)]
        axes.set_ylabel("Počet zpráv")
        legend = figure.legend(loc="outside right upper")
        figure.canvas.draw()
        assert tuple(line.get_color() for line in lines) == PALETTE
        assert axes.yaxis.label.get_fontfamily() == [TEST_FONT]
        assert axes.yaxis.label.get_fontsize() == FONT_SIZE
        assert not any(
            axes.spines[side].get_visible() for side in ("top", "right", "left")
        )
        assert axes.spines["bottom"].get_visible()
        assert all(line.get_visible() for line in axes.get_ygridlines())
        assert all(line.get_color() == "#DDDDDD" for line in axes.get_ygridlines())
        assert not any(line.get_visible() for line in axes.get_xgridlines())
        assert axes.get_axisbelow() is True
        assert not legend.get_frame_on()
        assert figure.get_constrained_layout()
        assert mpl.rcParams["pdf.fonttype"] == PDF_FONT_TYPE
        assert mpl.rcParams["savefig.pad_inches"] == 0
        assert mpl.rcParams["axes.formatter.use_locale"] is False
    assert figure.get_size_inches() * 2.54 == pytest.approx([16.5, 6])


def test_figure_restores_style_after_exception() -> None:
    """Ani přerušení kreslení nezmění globální rcParams.

    Raises:
        RuntimeError: Záměrné přerušení kreslení uvnitř testovaného kontextu.
    """
    before = mpl.rcParams.copy()
    msg = "přerušení"
    with (
        pytest.raises(RuntimeError, match="přerušení"),
        thesis_figure.figure(5, font=TEST_FONT),
    ):
        raise RuntimeError(msg)
    assert mpl.rcParams.copy() == before


@pytest.mark.parametrize(
    ("value", "expected"),
    [(12345, "12\u00a0345"), (1234.5, "1\u00a0234,5"), (-0.25, "-0,25")],
)
def test_number_axis(value: float, expected: str) -> None:
    """Osa čísel používá český zápis domény."""
    assert thesis_figure.number_axis()(value, 0) == expected


@pytest.mark.parametrize(
    ("value", "expected"), [(0.125, "12,5\u00a0%"), (0.07, "7\u00a0%"), (0, "0\u00a0%")]
)
def test_percent_axis(value: float, expected: str) -> None:
    """Procentní osa odstraní float artefakty a oddělí procenta NBSP."""
    assert thesis_figure.percent_axis()(value, 0) == expected


def test_sequential_colormap() -> None:
    """Obrácená viridis vynechá tmavý konec a skryje NA bílou."""
    colormap = thesis_figure.sequential_colormap()
    assert colormap(0.0) == pytest.approx(mpl.colormaps["viridis"](1.0))
    assert colormap(1.0) == pytest.approx(mpl.colormaps["viridis"](0.45))
    masked = np.ma.masked_invalid([0, float("nan"), 1])
    assert colormap(masked)[1] == pytest.approx(to_rgba("white"))


def test_save_pdf_dimensions_fonts_and_style(tmp_path: Path) -> None:
    """PDF zachová fyzický rozměr i písmo při uložení mimo kontext kreslení."""
    with thesis_figure.figure(6, font=TEST_FONT) as (figure, axes):
        axes.plot([0, 1], [0, 1], label="Řada")
        axes.set_title("Příliš žluťoučký kůň")
        figure.legend(loc="outside right upper")
    path = tmp_path / "figure.pdf"
    thesis_figure.save_pdf(figure, path, font=TEST_FONT)
    pdf = path.read_bytes()
    assert thesis_figure.embedded_fonts(pdf) == {"DejaVuSerif"}
    box = re.search(rb"/MediaBox\s*\[([^]]+)\]", pdf)
    assert box is not None
    assert [float(value) for value in box[1].split()] == pytest.approx([
        0,
        0,
        16.5 / 2.54 * 72,
        6 / 2.54 * 72,
    ])


def test_embedded_fonts_removes_subset_prefix() -> None:
    """Čtenář vrátí všechny řezy bez šestipísmenného prefixu podmnožiny."""
    pdf = b"/BaseFont /ABCDEF+LibertinusSerif-Regular /BaseFont /LibertinusSerif-Bold /BaseFont /UVWXYZ+DejaVuSans"
    assert thesis_figure.embedded_fonts(pdf) == {
        "LibertinusSerif-Regular",
        "LibertinusSerif-Bold",
        "DejaVuSans",
    }


def test_missing_font_fails_before_drawing() -> None:
    """Chybějící rodina nesmí potichu přejít na náhradní písmo."""
    with (
        pytest.raises(
            InvalidInputError,
            match=r"libertinus-fonts.*fonts-libertinus.*install-libertinus\.sh",
        ),
        thesis_figure.figure(5, font="Mediparse Missing Font"),
    ):
        pass


def test_missing_font_fails_before_saving(tmp_path: Path) -> None:
    """Chybějící písmo nevyprodukuje soubor s náhradní rodinou."""
    path = tmp_path / "missing.pdf"
    with pytest.raises(
        InvalidInputError,
        match=r"libertinus-fonts.*fonts-libertinus.*install-libertinus\.sh",
    ):
        thesis_figure.save_pdf(Figure(), path, font="Mediparse Missing Font")
    assert not path.exists()


@pytest.mark.parametrize("foreign_text", ["cizí písmo", "$x^2$"])
def test_foreign_font_removes_pdf(tmp_path: Path, foreign_text: str) -> None:
    """Odmítnutí cizí rodiny i příměsi mathtextu odstraní neplatný export."""
    with thesis_figure.figure(5, font=TEST_FONT) as (figure, axes):
        axes.set_title(foreign_text, fontfamily="DejaVu Sans")
    path = tmp_path / "foreign.pdf"
    with pytest.raises(InvalidInputError, match="DejaVuSans"):
        thesis_figure.save_pdf(figure, path, font=TEST_FONT)
    assert not path.exists()


def test_thesis_font_pdf(tmp_path: Path) -> None:
    """Skutečné písmo práce je povinné v jobu Test a jinde lze test přeskočit.

    Raises:
        InvalidInputError: Povinné písmo práce chybí.
    """
    try:
        thesis_figure.register_thesis_font()
    except InvalidInputError as error:
        if "MEDIPARSE_REQUIRE_THESIS_FONT" in os.environ:
            raise
        pytest.skip(str(error))
    count = len(font_manager.fontManager.ttflist)
    thesis_figure.register_thesis_font()
    assert len(font_manager.fontManager.ttflist) == count
    with thesis_figure.figure(5) as (figure, axes):
        axes.set_title("Český popisek")
        axes.set_ylabel("Počet")
    path = tmp_path / "thesis.pdf"
    thesis_figure.save_pdf(figure, path)
    fonts = thesis_figure.embedded_fonts(path.read_bytes())
    assert fonts
    assert all(name.startswith("LibertinusSerif-") for name in fonts)
