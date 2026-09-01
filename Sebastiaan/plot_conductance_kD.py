"""
Plot the conductance-variant kD sweep written by conductance_kD_runner.py.

Overlays one error-vs-kD curve per conductance scenario, star-marks each
variant's optimum kD. The shift of the star across variants IS the
kD<->conductance collinearity. Pure read -- iterate freely.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RUN_DIR = Path("../SavedData")     # folder with conductance_kD_sweep.nc, or its parent


def resolve(root):
    root = Path(root)
    if (root / "conductance_kD_sweep.nc").exists():
        return root
    subs = sorted(root.glob("conductance_kD_*"))
    if not subs:
        raise FileNotFoundError(f"No conductance_kD_sweep.nc or conductance_kD_* "
                                f"under {root.resolve()}")
    return subs[-1]


def main(run_dir):
    run_dir = resolve(run_dir)
    print(f"Reading from: {run_dir.resolve()}")
    ds = xr.open_dataset(run_dir / "conductance_kD_sweep.nc")
    kd = ds["kD"].values
    mae = ds["mae"].values
    labels = [str(v) for v in ds["variant"].values]
    gamma = float(ds.attrs.get("gamma", 0.0))

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = plt.cm.tab10(range(len(labels)))
    for i, label in enumerate(labels):
        line, = ax.plot(kd, mae[i], marker=".", color=colors[i], label=label)
        j = int(np.nanargmin(mae[i]))
        ax.plot(kd[j], mae[i][j], "*", ms=15, color=line.get_color(), zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel(r"kD (m$^2$/day)")
    ax.set_ylabel("mean absolute error (m)")
    ax.set_title("Error vs transmissivity, conductance variants "
                 f"($\\gamma$ = {gamma:g})\n"
                 "$\\star$ = optimum kD per variant (noise-free)")
    ax.legend(title="conductance")

    out = run_dir / "conductance_kD.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(run_dir / "conductance_kD.pdf", bbox_inches="tight")
    print(f"Saved {out}")
    return fig


if __name__ == "__main__":
    main(RUN_DIR)
    plt.show()
