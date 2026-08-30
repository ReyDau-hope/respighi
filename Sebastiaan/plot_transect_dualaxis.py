"""
Dual-axis transect: head (left axis, blue) and recharge (right axis, red) on the
same plot, one subplot per featured case. Quick, compact "good fit / bad
recharge" demonstration -- one subplot per case so a good and a bad case sit
side by side.

CAUTION (built in as honestly as possible): twin y-axes let the axis scaling
distort the apparent relationship. To keep it honest, the recharge axis is
forced SYMMETRIC about zero with a bold zero line, so "recharge goes negative"
is read from the physics (below zero), not from arbitrary scaling. State the
recharge axis range in your caption.

Works on a kD or reg sweep folder (auto-detected). Set FEATURE_VALUES; None =
lowest and highest swept value.
"""
#%%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RUN_DIR = Path(r"C:\Users\sebas\Documents\1Thesis\respighi\SavedData\transect_reg_20260826_2106")
TRANSECT_FRAC = 0.5
FEATURE_VALUES = [100, 4000]              # e.g. [4000, 100]; None = extremes

_PAT = re.compile(r"head_([A-Za-z]+)(\d+)\.nc$")

C_HEAD = "tab:blue"
C_RECH = "tab:red"


def resolve(root):
    root = Path(root)
    if any(root.glob("head_*.nc")):
        return root
    subs = sorted(root.glob("transect_*"))
    if not subs:
        raise FileNotFoundError(f"No head_*.nc or transect_* under {root.resolve()}")
    return subs[-1]


def transect_line(da2d, frac):
    y = da2d["y"].values
    y_pick = y.min() + frac * (y.max() - y.min())
    line = da2d.sel(y=y_pick, method="nearest")
    return line["x"].values, line.values, float(line["y"].values)


def sweep_and_values(run_dir):
    files = sorted(run_dir.glob("head_*.nc"))
    if not files:
        raise FileNotFoundError("No head_*.nc files.")
    sweep = _PAT.search(files[0].name).group(1)
    vals = sorted(int(_PAT.search(f.name).group(2)) for f in files)
    return sweep, vals


def find_file(run_dir, sweep, value):
    p = run_dir / f"head_{sweep}{int(round(value)):05d}.nc"
    if p.exists():
        return p
    for f in run_dir.glob("head_*.nc"):
        m = _PAT.search(f.name)
        if m and m.group(1) == sweep and int(m.group(2)) == int(round(value)):
            return f
    raise FileNotFoundError(f"No file for {sweep}={value}")


def main(run_dir):
    run_dir = resolve(run_dir)
    print(f"Reading from: {run_dir.resolve()}")
    sweep, all_vals = sweep_and_values(run_dir)
    feats = FEATURE_VALUES if FEATURE_VALUES else [all_vals[0], all_vals[-1]]

    truth = xr.open_dataset(run_dir / "truth.nc")["head"]
    xt, ht, y_used = transect_line(truth, TRANSECT_FRAC)

    fig, axes = plt.subplots(1, len(feats), figsize=(6.5 * len(feats), 4.8),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]

    for ax, v in zip(axes, feats):
        ds = xr.open_dataset(find_file(run_dir, sweep, v))
        x, h, _ = transect_line(ds["head"], TRANSECT_FRAC)
        _, r, _ = transect_line(ds["recharge"], TRANSECT_FRAC)

        # left axis: head (+ truth)
        ax.plot(xt, ht, color="black", lw=2.5, label="truth", zorder=5)
        ax.plot(x, h, color=C_HEAD, lw=1.8, label="fitted head")
        ax.set_xlabel("distance along transect (x, m)")
        ax.set_ylabel("head (m)", color=C_HEAD)
        ax.tick_params(axis="y", labelcolor=C_HEAD)
        ax.set_title(f"{sweep} = {v:g}")

        # right axis: recharge, symmetric about zero
        ax2 = ax.twinx()
        ax2.plot(x, r, color=C_RECH, lw=1.6, alpha=0.9, label="fitted recharge")
        ax2.axhline(0, color=C_RECH, ls=":", lw=1.0)
        rmax = float(np.nanmax(np.abs(r)))
        ax2.set_ylim(-1.1 * rmax, 1.1 * rmax)
        ax2.set_ylabel("recharge (m/day)", color=C_RECH)
        ax2.tick_params(axis="y", labelcolor=C_RECH)

    fig.suptitle(f"Head (blue, left) vs. recharge (red, right) along a transect "
                 f"(y = {y_used:.0f} m)\nrecharge axis symmetric about zero", fontsize=12)

    out = run_dir / f"transect_dualaxis_{sweep}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(run_dir / f"transect_dualaxis_{sweep}.pdf", bbox_inches="tight")
    print(f"Saved {out}")
    return fig


if __name__ == "__main__":
    main(RUN_DIR)
    plt.show()

# %%
