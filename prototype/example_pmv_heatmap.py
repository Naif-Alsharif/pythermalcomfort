"""Simple PMV diverging heatmap example using RangeScene."""

from __future__ import annotations

import matplotlib.pyplot as plt
from range_scene import RangeScene

from pythermalcomfort.models import pmv_ppd_iso


def main() -> None:
    # 1) Create a scene with the PMV model.
    scene = RangeScene(model_func=pmv_ppd_iso)

    # 2) Configure the grid and model inputs using dot-notation chaining.
    scene.axes(
        x_name="tdb",
        x_min=18.0,
        x_max=34.5,
        x_step=0.5,
        y_name="rh",
        y_min=20.0,
        y_max=100.0,
        y_step=2.0,
    ).link(
        tr="tdb"  # Keep tr equal to tdb at each grid point.
    ).parameters(
        vr=0.10,
        met=1.2,
        clo=0.5,
        wme=0.0,
        limit_inputs=False,
        round_output=False,
    ).metric("pmv")

    # 3) Plot a diverging PMV heatmap with PMV comfort contours.
    ax = scene.plot(
        cmap="coolwarm",
        diverging_center=0.0,
        contour_levels=[-0.5, 0.5],
        contour_colors=["#08306b", "#7f0000"],
    )

    # 4) Add beginner-friendly labels and note.
    ax.set_title("PMV Diverging Heatmap over Air Temperature and Relative Humidity")
    ax.set_xlabel("Air temperature [degC]")
    ax.set_ylabel("Relative humidity [%]")
    ax.text(
        0.01,
        -0.12,
        "Fixed inputs: tr=tdb, vr=0.10 m/s, met=1.2, clo=0.5",
        transform=ax.transAxes,
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
