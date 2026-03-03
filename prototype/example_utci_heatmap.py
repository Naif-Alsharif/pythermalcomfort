"""Simple UTCI heatmap example using RangeScene."""

from __future__ import annotations

import matplotlib.pyplot as plt
from range_scene import RangeScene

from pythermalcomfort.models import utci


def main() -> None:
    # 1) Create a scene with the UTCI model.
    scene = RangeScene(model_func=utci)

    # 2) Configure grid and UTCI assumptions using dot-notation chaining.
    scene.axes(
        x_name="tdb",
        x_min=-10.0,
        x_max=40.0,
        x_step=1.0,
        y_name="rh",
        y_min=20.0,
        y_max=100.0,
        y_step=2.0,
    ).link(
        tr="tdb"  # Simple assumption: mean radiant temperature equals air temperature.
    ).parameters(
        v=1.0,  # Wind speed in m/s (UTCI uses v, not vr).
        units="SI",
        limit_inputs=True,
        round_output=False,
    ).metric("utci")

    # 3) Plot heatmap and add a few UTCI threshold contours.
    ax = scene.plot(
        cmap="coolwarm",
        contour_levels=[26.0, 32.0],
        contour_colors=["#238443", "#f16913"],
    )

    # 4) Add beginner-friendly labels.
    ax.set_title("UTCI Heatmap over Air Temperature and Relative Humidity")
    ax.set_xlabel("Air temperature [degC]")
    ax.set_ylabel("Relative humidity [%]")
    ax.text(
        0.01,
        -0.12,
        "Assumptions: tr=tdb, v=1.0 m/s, SI units",
        transform=ax.transAxes,
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
