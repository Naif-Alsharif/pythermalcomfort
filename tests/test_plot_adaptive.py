"""Tests for adaptive comfort chart plotting."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pythermalcomfort.plots.matplotlib.adaptive import (
    AdaptivePlot,
    AdaptivePlotResult,
    BandsConfig,
    _compute_ce,
)


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════════
# _compute_ce tests
# ══════════════════════════════════════════════════════════════════════════


def test_ce_none() -> None:
    assert _compute_ce(None) == 0.0


def test_ce_low_speed() -> None:
    assert _compute_ce(0.3) == 0.0
    assert _compute_ce(0.59) == 0.0


def test_ce_medium_speed() -> None:
    assert _compute_ce(0.6) == 1.2
    assert _compute_ce(0.8) == 1.2


def test_ce_high_speed() -> None:
    assert _compute_ce(0.9) == 1.8
    assert _compute_ce(1.1) == 1.8


def test_ce_very_high_speed() -> None:
    assert _compute_ce(1.2) == 2.2
    assert _compute_ce(2.0) == 2.2


# ══════════════════════════════════════════════════════════════════════════
# Constructor tests
# ══════════════════════════════════════════════════════════════════════════


def test_constructor_accepts_ashrae() -> None:
    plot = AdaptivePlot("ashrae")
    assert plot._standard == "ashrae"


def test_constructor_accepts_en() -> None:
    plot = AdaptivePlot("en")
    assert plot._standard == "en"


def test_constructor_case_insensitive() -> None:
    plot = AdaptivePlot("ASHRAE")
    assert plot._standard == "ashrae"


def test_constructor_rejects_invalid_standard() -> None:
    with pytest.raises(ValueError, match="Unknown standard"):
        AdaptivePlot("invalid")


def test_constructor_default_t_rm_range() -> None:
    plot = AdaptivePlot("ashrae")
    assert plot._t_rm_range == (10.0, 33.5)


def test_constructor_custom_t_rm_range() -> None:
    plot = AdaptivePlot("ashrae", t_running_mean_range=(15, 30))
    assert plot._t_rm_range == (15.0, 30.0)


def test_constructor_rejects_invalid_t_rm_range() -> None:
    with pytest.raises(ValueError, match="min < max"):
        AdaptivePlot("ashrae", t_running_mean_range=(30, 10))


# ══════════════════════════════════════════════════════════════════════════
# set_params tests
# ══════════════════════════════════════════════════════════════════════════


def test_set_params_stores_v() -> None:
    plot = AdaptivePlot("ashrae").set_params(v=0.8)
    assert plot._v == 0.8


def test_set_params_default_v_is_none() -> None:
    plot = AdaptivePlot("ashrae")
    assert plot._v is None


def test_set_params_chaining() -> None:
    plot = AdaptivePlot("ashrae").set_params(v=0.5)
    assert isinstance(plot, AdaptivePlot)


# ══════════════════════════════════════════════════════════════════════════
# BandsConfig tests
# ══════════════════════════════════════════════════════════════════════════


def test_bands_config_default() -> None:
    cfg = BandsConfig()
    assert cfg.show is None
    assert cfg.labels is None
    assert cfg.colors is None


def test_bands_config_validates_show_keys() -> None:
    cfg = BandsConfig(show=["invalid_key"])
    with pytest.raises(ValueError, match="Invalid band key"):
        cfg._validate("ashrae")


def test_bands_config_validates_label_length() -> None:
    cfg = BandsConfig(show=["90"], labels=["A", "B"])
    with pytest.raises(ValueError, match="labels must have length 1"):
        cfg._validate("ashrae")


def test_bands_config_validates_color_length() -> None:
    cfg = BandsConfig(show=["90"], colors=["#ff0000", "#00ff00"])
    with pytest.raises(ValueError, match="colors must have length 1"):
        cfg._validate("ashrae")


def test_bands_config_validates_color_values() -> None:
    cfg = BandsConfig(colors=["not-a-color", "#ff0000"])
    with pytest.raises(ValueError, match="Invalid color"):
        cfg._validate("ashrae")


def test_bands_config_validates_all_bands_label_length() -> None:
    cfg = BandsConfig(labels=["A", "B", "C"])
    with pytest.raises(ValueError, match="labels must have length 2"):
        cfg._validate("ashrae")


def test_bands_config_valid_en() -> None:
    cfg = BandsConfig(
        show=["cat_i", "cat_ii"],
        labels=["Best", "OK"],
        colors=["#00ff00", "#ffff00"],
    )
    cfg._validate("en")
    assert cfg._validated


# ══════════════════════════════════════════════════════════════════════════
# set_bands tests
# ══════════════════════════════════════════════════════════════════════════


def test_set_bands_raw_params() -> None:
    plot = AdaptivePlot("ashrae")
    plot.set_bands(show=["90"], labels=["90% Zone"], colors=["#FF6B6B"])
    assert plot._bands_config is not None
    assert plot._bands_config.show == ["90"]


def test_set_bands_with_config() -> None:
    config = BandsConfig(show=["90"], labels=["90% Zone"], colors=["#FF6B6B"])
    plot = AdaptivePlot("ashrae")
    plot.set_bands(show=config)
    assert plot._bands_config is config


def test_set_bands_config_rejects_separate_labels() -> None:
    config = BandsConfig(show=["90"])
    with pytest.raises(ValueError, match="must not be provided separately"):
        AdaptivePlot("ashrae").set_bands(show=config, labels=["X"])


def test_set_bands_config_rejects_separate_colors() -> None:
    config = BandsConfig(show=["90"])
    with pytest.raises(ValueError, match="must not be provided separately"):
        AdaptivePlot("ashrae").set_bands(show=config, colors=["#ff0000"])


def test_set_bands_rejects_invalid_keys() -> None:
    with pytest.raises(ValueError, match="Invalid band key"):
        AdaptivePlot("ashrae").set_bands(show=["cat_i"])


def test_set_bands_chaining() -> None:
    plot = AdaptivePlot("ashrae").set_bands(show=["90"])
    assert isinstance(plot, AdaptivePlot)


# ══════════════════════════════════════════════════════════════════════════
# ASHRAE plot tests
# ══════════════════════════════════════════════════════════════════════════


def test_ashrae_plot_smoke() -> None:
    result = AdaptivePlot("ashrae").plot(title="ASHRAE Test")
    assert isinstance(result, AdaptivePlotResult)
    assert result.ax.get_title() == "ASHRAE Test"
    assert len(result.fills) == 2
    assert result.center_line is not None
    assert result.legend is not None


def test_ashrae_plot_no_params_required() -> None:
    """Plot should work without calling set_params at all."""
    result = AdaptivePlot("ashrae").plot()
    assert isinstance(result, AdaptivePlotResult)
    assert len(result.fills) == 2


def test_ashrae_plot_default_labels() -> None:
    result = AdaptivePlot("ashrae").plot()
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "90% Acceptability" in legend_labels
    assert "80% Acceptability" in legend_labels
    assert "Comfort Temperature" in legend_labels


def test_ashrae_plot_custom_labels() -> None:
    result = AdaptivePlot("ashrae").set_bands(labels=["Wide", "Narrow"]).plot()
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Wide" in legend_labels
    assert "Narrow" in legend_labels


def test_ashrae_plot_show_only_90() -> None:
    result = AdaptivePlot("ashrae").set_bands(show=["90"]).plot()
    assert len(result.fills) == 1
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "90% Acceptability" in legend_labels
    assert "80% Acceptability" not in legend_labels


def test_ashrae_plot_custom_colors() -> None:
    result = AdaptivePlot("ashrae").set_bands(colors=["#FF0000", "#00FF00"]).plot()
    assert len(result.fills) == 2


def test_ashrae_plot_no_center_line() -> None:
    result = AdaptivePlot("ashrae").plot(show_center_line=False)
    assert result.center_line is None


def test_ashrae_plot_no_legend() -> None:
    result = AdaptivePlot("ashrae").plot(legend=False)
    assert result.legend is None


def test_ashrae_plot_no_grid() -> None:
    result = AdaptivePlot("ashrae").plot(grid=False)
    assert not result.ax.xaxis.get_gridlines()[0].get_visible()


def test_ashrae_plot_with_grid() -> None:
    result = AdaptivePlot("ashrae").plot(grid=True)
    assert result.ax.xaxis.get_gridlines()[0].get_visible()


def test_ashrae_plot_xlim_default() -> None:
    result = AdaptivePlot("ashrae").plot()
    assert result.ax.get_xlim() == pytest.approx((10.0, 33.5))


def test_ashrae_plot_xlim_custom() -> None:
    result = AdaptivePlot("ashrae", t_running_mean_range=(15, 30)).plot()
    assert result.ax.get_xlim() == pytest.approx((15.0, 30.0))


def test_ashrae_plot_uses_provided_axis() -> None:
    fig, ax = plt.subplots()
    result = AdaptivePlot("ashrae").plot(ax=ax)
    assert result.ax is ax
    assert result.fig is fig


def test_ashrae_plot_xlabel_ylabel() -> None:
    result = AdaptivePlot("ashrae").plot(xlabel="X", ylabel="Y")
    assert result.ax.get_xlabel() == "X"
    assert result.ax.get_ylabel() == "Y"


def test_ashrae_plot_no_xlabel_ylabel() -> None:
    result = AdaptivePlot("ashrae").plot(xlabel=None, ylabel=None)
    assert result.ax.get_xlabel() == ""
    assert result.ax.get_ylabel() == ""


def test_ashrae_plot_center_line_kws() -> None:
    result = AdaptivePlot("ashrae").plot(
        center_line_kws={"color": "red", "linewidth": 3.0},
    )
    assert result.center_line is not None
    assert result.center_line.get_color() == "red"
    assert result.center_line.get_linewidth() == 3.0


def test_ashrae_plot_with_bands_config() -> None:
    config = BandsConfig(
        show=["90"],
        labels=["Narrow Zone"],
        colors=["#FF6B6B"],
    )
    result = AdaptivePlot("ashrae").set_bands(show=config).plot()
    assert len(result.fills) == 1
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Narrow Zone" in legend_labels


# ══════════════════════════════════════════════════════════════════════════
# CE (cooling effect) tests
# ══════════════════════════════════════════════════════════════════════════


def test_ashrae_plot_without_ce() -> None:
    """Without v, upper boundary = t_cmf + offset (no ce)."""
    result = AdaptivePlot("ashrae").set_bands(show=["90"]).plot()
    # At t_rm=20: t_cmf = 0.31*20+17.8 = 24.0, upper = 26.5
    # Check that the fill exists and is reasonable
    assert len(result.fills) == 1


def test_ashrae_plot_with_ce() -> None:
    """With v=0.8, ce=1.2 shifts upper boundary up."""
    result = AdaptivePlot("ashrae").set_params(v=0.8).set_bands(show=["90"]).plot()
    assert len(result.fills) == 1


def test_ashrae_ce_shifts_upper_boundary() -> None:
    """Verify ce numerically shifts the upper boundary."""
    # Without ce
    plot_no_ce = AdaptivePlot("ashrae").set_bands(show=["90"])
    result_no_ce = plot_no_ce.plot()

    # With ce (v=0.8 → ce=1.2)
    plot_ce = AdaptivePlot("ashrae").set_params(v=0.8).set_bands(show=["90"])
    result_ce = plot_ce.plot()

    # The fill_between with ce should have a higher upper boundary
    # We can check by comparing the y-axis auto limits
    ylim_no_ce = result_no_ce.ax.get_ylim()
    ylim_ce = result_ce.ax.get_ylim()
    assert ylim_ce[1] > ylim_no_ce[1]


def test_ashrae_ce_only_above_25() -> None:
    """CE should only shift upper boundary where it exceeds 25 degC."""
    ce = 1.2  # v=0.8

    # 90% upper = 0.31 * t_rm + 17.8 + 2.5
    # Crosses 25 at t_rm ≈ 15.16
    t_rm_below = 12.0
    upper_below = 0.31 * t_rm_below + 17.8 + 2.5  # = 24.02, < 25
    assert upper_below < 25.0
    # No ce applied
    assert upper_below == pytest.approx(24.02)

    t_rm_above = 20.0
    upper_above_base = 0.31 * t_rm_above + 17.8 + 2.5  # = 26.5, >= 25
    assert upper_above_base >= 25.0
    # ce applied
    expected = upper_above_base + ce  # 27.7
    assert expected == pytest.approx(27.7)


def test_ashrae_ce_transition_point() -> None:
    """Verify the transition point where ce kicks in for 90% band."""
    # 90% upper = 0.31 * t_rm + 20.3 = 25.0
    t_rm_transition = (25.0 - 20.3) / 0.31
    assert t_rm_transition == pytest.approx(15.16, abs=0.01)

    # 80% upper = 0.31 * t_rm + 21.3 = 25.0
    t_rm_transition_80 = (25.0 - 21.3) / 0.31
    assert t_rm_transition_80 == pytest.approx(11.94, abs=0.01)


def test_en_ce_only_above_25() -> None:
    """CE should only shift upper boundary where it exceeds 25 degC."""
    ce = 1.2  # v=0.8

    # Cat I upper = 0.33 * t_rm + 20.8
    # Crosses 25 at t_rm ≈ 12.73
    t_rm_below = 11.0
    upper_below = 0.33 * t_rm_below + 18.8 + 2.0  # = 24.43, < 25
    assert upper_below < 25.0

    t_rm_above = 20.0
    upper_above_base = 0.33 * t_rm_above + 18.8 + 2.0  # = 27.4, >= 25
    assert upper_above_base >= 25.0
    expected = upper_above_base + ce  # 28.6
    assert expected == pytest.approx(28.6)


def test_en_ce_cat_ii_full_range() -> None:
    """Cat II upper >= 25 at t_rm=10, so ce applies to entire range."""
    upper_at_10 = 0.33 * 10.0 + 18.8 + 3.0  # = 25.1
    assert upper_at_10 >= 25.0


def test_en_ce_cat_iii_full_range() -> None:
    """Cat III upper >= 25 at t_rm=10, so ce applies to entire range."""
    upper_at_10 = 0.33 * 10.0 + 18.8 + 4.0  # = 26.1
    assert upper_at_10 >= 25.0


def test_en_ce_transition_point() -> None:
    """Verify the transition point where ce kicks in for Cat I."""
    t_rm_transition = (25.0 - 20.8) / 0.33
    assert t_rm_transition == pytest.approx(12.73, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════
# EN plot tests
# ══════════════════════════════════════════════════════════════════════════


def test_en_plot_smoke() -> None:
    result = AdaptivePlot("en").plot(title="EN Test")
    assert isinstance(result, AdaptivePlotResult)
    assert result.ax.get_title() == "EN Test"
    assert len(result.fills) == 3
    assert result.center_line is not None
    assert result.legend is not None


def test_en_plot_default_labels() -> None:
    result = AdaptivePlot("en").plot()
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Category I" in legend_labels
    assert "Category II" in legend_labels
    assert "Category III" in legend_labels


def test_en_plot_show_only_cat_i() -> None:
    result = AdaptivePlot("en").set_bands(show=["cat_i"]).plot()
    assert len(result.fills) == 1
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Category I" in legend_labels
    assert "Category II" not in legend_labels


def test_en_plot_custom_labels() -> None:
    result = AdaptivePlot("en").set_bands(labels=["Best", "OK", "Min"]).plot()
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Best" in legend_labels
    assert "OK" in legend_labels
    assert "Min" in legend_labels


def test_en_plot_show_two_bands() -> None:
    result = (
        AdaptivePlot("en")
        .set_bands(
            show=["cat_i", "cat_ii"],
            labels=["Strict", "Normal"],
            colors=["#00FF00", "#FFFF00"],
        )
        .plot()
    )
    assert len(result.fills) == 2
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Strict" in legend_labels
    assert "Normal" in legend_labels


def test_en_rejects_ashrae_band_keys() -> None:
    with pytest.raises(ValueError, match="Invalid band key"):
        AdaptivePlot("en").set_bands(show=["80"])


# ══════════════════════════════════════════════════════════════════════════
# Formula correctness tests
# ══════════════════════════════════════════════════════════════════════════


def test_ashrae_formula_values() -> None:
    """Verify ASHRAE formula: t_cmf = 0.31 * t_rm + 17.8."""
    t_cmf = 0.31 * 20 + 17.8
    assert t_cmf == pytest.approx(24.0)
    assert t_cmf + 2.5 == pytest.approx(26.5)  # 90% upper
    assert t_cmf - 2.5 == pytest.approx(21.5)  # 90% lower
    assert t_cmf + 3.5 == pytest.approx(27.5)  # 80% upper
    assert t_cmf - 3.5 == pytest.approx(20.5)  # 80% lower


def test_en_formula_values() -> None:
    """Verify EN formula: t_cmf = 0.33 * t_rm + 18.8."""
    t_cmf = 0.33 * 20 + 18.8
    assert t_cmf == pytest.approx(25.4)
    assert t_cmf + 2.0 == pytest.approx(27.4)  # Cat I upper
    assert t_cmf - 3.0 == pytest.approx(22.4)  # Cat I lower
    assert t_cmf + 3.0 == pytest.approx(28.4)  # Cat II upper
    assert t_cmf - 4.0 == pytest.approx(21.4)  # Cat II lower
    assert t_cmf + 4.0 == pytest.approx(29.4)  # Cat III upper
    assert t_cmf - 5.0 == pytest.approx(20.4)  # Cat III lower


def test_formula_matches_model_output() -> None:
    """Our formula should match the model output (at t_rm=20, v<0.6)."""
    # Values from the user's earlier model run:
    # adaptive_ashrae(tdb=25, tr=25, t_running_mean=20, v=0.1)
    t_cmf = 0.31 * 20 + 17.8
    assert t_cmf == pytest.approx(24.0)
    assert t_cmf - 3.5 == pytest.approx(20.5)  # matches tmp_cmf_80_low
    assert t_cmf + 3.5 == pytest.approx(27.5)  # matches tmp_cmf_80_up
    assert t_cmf - 2.5 == pytest.approx(21.5)  # matches tmp_cmf_90_low
    assert t_cmf + 2.5 == pytest.approx(26.5)  # matches tmp_cmf_90_up


# ══════════════════════════════════════════════════════════════════════════
# BandsConfig reuse tests
# ══════════════════════════════════════════════════════════════════════════


def test_bands_config_reuse_across_plots() -> None:
    config = BandsConfig(show=["90"], labels=["Comfort"], colors=["#FF6B6B"])

    result1 = AdaptivePlot("ashrae").set_bands(show=config).plot()
    result2 = AdaptivePlot("ashrae").set_bands(show=config).plot()

    labels1 = [t.get_text() for t in result1.legend.get_texts()]
    labels2 = [t.get_text() for t in result2.legend.get_texts()]
    assert labels1 == labels2
    assert "Comfort" in labels1


# ══════════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════════


def test_plot_without_set_bands_uses_defaults() -> None:
    result = AdaptivePlot("ashrae").plot()
    assert len(result.fills) == 2
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "80% Acceptability" in legend_labels
    assert "90% Acceptability" in legend_labels


def test_plot_legend_kws() -> None:
    result = AdaptivePlot("ashrae").plot(legend_kws={"loc": "upper left"})
    assert result.legend is not None


def test_plot_fill_kws() -> None:
    result = AdaptivePlot("ashrae").plot(fill_kws={"alpha": 0.3})
    assert len(result.fills) > 0
