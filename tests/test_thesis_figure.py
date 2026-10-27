"""Chování stylu grafů, českých os a barevných škál práce."""

from __future__ import annotations

import os
import re
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import thesis_figure

TEST_FONT = "DejaVu Serif"
PDF_FONT_TYPE = 3


def test_figure_dimensions_style_and_legend() -> None:
    """Graf má sazební rozměr, paletu, mřížku a legendu vně vpravo."""
    with thesis_figure.figure(6, font=TEST_FONT) as (figure, axes):
        lines = [
            axes.plot([0, 1], [0, i], label=str(i))[0]
            for i in range(len(thesis_figure.PALETTE_CATEGORICAL))
        ]
        axes.set_ylabel("Počet zpráv")
        legend = figure.legend(loc="outside right upper")
        figure.canvas.draw()
        assert (
            tuple(line.get_color() for line in lines)
            == thesis_figure.PALETTE_CATEGORICAL
        )
        assert axes.yaxis.label.get_fontfamily() == [TEST_FONT]
        assert axes.yaxis.label.get_fontsize() == thesis_figure.THESIS_FONT_SIZE
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
    assert figure.get_size_inches() * 2.54 == pytest.approx([
        thesis_figure.THESIS_WIDTH_CM,
        6,
    ])


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
    assert _luma(colormap(0.0)) > _luma(colormap(1.0))
    assert _lightness(colormap(1.0)) >= MIN_LIGHTNESS
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
        thesis_figure.THESIS_WIDTH_CM / 2.54 * 72,
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


def test_shared_subplots_keep_pdf_dimensions(tmp_path: Path) -> None:
    """Sdílené osy i více panelů zachovají rozměr stránky pro práci."""
    with thesis_figure.figure(
        7.5, nrows=2, ncols=2, sharex=True, sharey=True, font=TEST_FONT
    ) as (figure, axes):
        axes[0, 0].plot([0, 1], [0, 1])
        assert axes.shape == (2, 2)
        assert axes[0, 0].get_shared_x_axes().joined(axes[0, 0], axes[1, 1])
        assert axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[1, 1])
    path = tmp_path / "panels.pdf"
    thesis_figure.save_pdf(figure, path, font=TEST_FONT)
    box = re.search(rb"/MediaBox\s*\[([^]]+)\]", path.read_bytes())
    assert box is not None
    assert [float(value) for value in box[1].split()] == pytest.approx(
        [0, 0, 467.7165354, 7.5 / 2.54 * 72], abs=0.01
    )


def test_registration_refreshes_stale_font_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registrace doplní rodinu nainstalovanou až po vzniku cache a je idempotentní."""
    path = str(Path(mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif.ttf")
    manager = font_manager.FontManager()
    manager.ttflist = [entry for entry in manager.ttflist if entry.name != TEST_FONT]
    properties = font_manager.FontProperties(family=TEST_FONT)
    with pytest.raises(ValueError, match="Failed to find font"):
        manager.findfont(properties, fallback_to_default=False)
    monkeypatch.setattr(font_manager, "fontManager", manager)
    monkeypatch.setattr(font_manager, "findSystemFonts", lambda: [path])
    thesis_figure.register_thesis_font(TEST_FONT)
    assert manager.findfont(properties, fallback_to_default=False) == path
    count = len(manager.ttflist)
    thesis_figure.register_thesis_font(TEST_FONT)
    assert len(manager.ttflist) == count


MIN_CATEGORICAL_COLORS = 5
MIN_LUMA_GAP = 0.12
OKABE_ITO = frozenset({
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
})


def _luma(color: str | tuple[float, float, float, float]) -> float:
    return sum(
        weight * channel
        for weight, channel in zip(
            (0.299, 0.587, 0.114), to_rgba(color)[:3], strict=True
        )
    )


def test_categorical_palette_is_distinct_in_grayscale() -> None:
    """Paleta zůstává podmnožinou Okabe-Ito rozlišitelnou i proti bílé v šedi."""
    palette = thesis_figure.PALETTE_CATEGORICAL
    assert set(palette) <= OKABE_ITO
    assert len(palette) >= MIN_CATEGORICAL_COLORS
    assert all(
        abs(_luma(first) - _luma(second)) >= MIN_LUMA_GAP
        for first, second in combinations((*palette, "#FFFFFF"), 2)
    )


MIN_LIGHTNESS = 49
SRGB_LINEAR_THRESHOLD = 0.04045
LIGHTNESS_PIVOT = (6 / 29) ** 3


def _lightness(color: tuple[float, float, float, float]) -> float:
    linear = tuple(
        channel / 12.92
        if channel <= SRGB_LINEAR_THRESHOLD
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in color[:3]
    )
    luminance = sum(
        weight * channel
        for weight, channel in zip((0.2126, 0.7152, 0.0722), linear, strict=True)
    )
    return (
        116 * luminance ** (1 / 3) - 16
        if luminance > LIGHTNESS_PIVOT
        else (29 / 3) ** 3 * luminance
    )
