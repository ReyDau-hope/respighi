"""
Replot a saved kD sweep from kD_sweep.nc (no re-solving).

Plots all solves + marks non-converged, and scales the y-axis to the CONVERGED
points so one non-converged spike doesn't flatten the real signal.
"""
#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RUN_DIR = Path(r"C:\Users\sebas\Documents\1Thesis\respighi\SavedData\kD_sweep_original_20260902_1300")     # folder with kD_sweep.nc, or its parent


def resolve(root):
    root = Path(root)
    if (root / "kD_sweep.nc").exists():
        return root
    subs = sorted(root.glob("kD_sweep_*"))
    if not subs:
        raise FileNotFoundError(f"No kD_sweep.nc or kD_sweep_* under {root.resolve()}")
    return subs[-1]


def main(run_dir):
    run_dir = resolve(run_dir)
    print(f"Reading from: {run_dir.resolve()}")
    ds = xr.open_dataset(run_dir / "kD_sweep.nc")
    kd = ds["kD"].values
    mae = ds["mae"].values
    converged = ds["converged"].values.astype(bool)
    gamma = float(ds.attrs.get("gamma", 0.0))
    scenario = ds.attrs.get("scenario", "")

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(kd, mae, marker=".", color="tab:blue", label="all solves", zorder=1)
    bad = ~converged
    if bad.any():
        ax.plot(kd[bad], mae[bad], "x", color="red", ms=9,
                label="not converged", zorder=3)

    # y-axis scaled to converged points
    good = converged
    if good.any():
        lo = float(np.nanmin(mae[good]))
        hi = float(np.nanmax(mae[good]))
        pad = 0.1 * (hi - lo) if hi > lo else 0.05
        ax.set_ylim(lo - pad, hi + pad)
        n_clipped = int(np.sum(mae[bad] > hi + pad)) if bad.any() else 0
        if n_clipped:
            ax.text(0.98, 0.98, f"{n_clipped} non-converged point(s) clipped above",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="red")

    ax.set_xscale("log")
    ax.set_xlabel(r"kD (m$^2$/day)")
    ax.set_ylabel("mean absolute error (m)")
    ax.set_title(f"Error vs transmissivity  ($\\gamma$ = {gamma:g}, {scenario})")
    ax.legend()

    out = run_dir / "kD_sweep_rescaled.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(run_dir / "kD_sweep_rescaled.pdf", bbox_inches="tight")
    print(f"Saved {out}")

    n_bad = int(bad.sum())
    print(f"{good.sum()}/{len(kd)} converged"
          + (f"  ({n_bad} not)" if n_bad else ""))
    i_all = int(np.nanargmin(mae))
    print(f"Lowest MAE overall  : kD = {kd[i_all]:.0f}, MAE = {mae[i_all]:.4f}, "
          f"converged = {bool(converged[i_all])}")
    if good.any():
        j = int(np.nanargmin(np.where(good, mae, np.inf)))
        print(f"Lowest MAE converged: kD = {kd[j]:.0f}, MAE = {mae[j]:.4f}")
    return fig


if __name__ == "__main__":
    main(RUN_DIR)
    plt.show()
