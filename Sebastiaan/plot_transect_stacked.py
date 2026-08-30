"""
Stacked-panel transect: head (top, with IBRAHYM truth) and recharge (bottom,
with zero line), sharing the same x-axis, for one or more featured cases.

The point: the head panel shows the fits hugging the truth (good fit at every
value), while the recharge panel shows how wildly the recharge differs to
achieve that fit -- and where it goes NEGATIVE (below the zero line), which is
physically implausible. Same head, very different recharge = the head can't tell
good from bad; the recharge can.

Works on a kD sweep or a reg sweep folder (auto-detected). Set FEATURE_VALUES to
the cases you want (e.g. a good and a bad one); leave it None to auto-pick the
lowest and highest swept value.
"""
#%%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RUN_DIR = Path(r"C:\Users\sebas\Documents\1Thesis\respighi\SavedData\transect_reg_20260826_2106")     # transect_* folder, or its parent
TRANSECT_FRAC = 0.5
FEATURE_VALUES = [100, 4000]              # e.g. [4000, 100] for reg, [2000, 250] for kD; None = extremes

_PAT = re.compile(r"head_([A-Za-z]+)(\d+)\.nc$")


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

    fig, (ax_h, ax_r) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                     constrained_layout=True)
    ax_h.plot(xt, ht, color="black", lw=3, label="IBRAHYM (truth)", zorder=10)

    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(feats)))
    rmax = 0.0
    for v, c in zip(feats, colors):
        ds = xr.open_dataset(find_file(run_dir, sweep, v))
        x, h, _ = transect_line(ds["head"], TRANSECT_FRAC)
        _, r, _ = transect_line(ds["recharge"], TRANSECT_FRAC)
        ax_h.plot(x, h, color=c, lw=1.8, label=f"{sweep} = {v:g}")
        ax_r.plot(x, r, color=c, lw=1.8, label=f"{sweep} = {v:g}")
        rmax = max(rmax, float(np.nanmax(np.abs(r))))

    ax_r.axhline(0, color="gray", lw=1.0, zorder=0)
    ax_r.set_ylim(-1.1 * rmax, 1.1 * rmax)          # symmetric so sign is honest

    ax_h.set_ylabel("head (m)")
    ax_h.set_title(f"Head fit vs. recharge along a transect  (y = {y_used:.0f} m)")
    ax_h.legend(fontsize=9)
    ax_r.set_ylabel("fitted recharge (m/day)")
    ax_r.set_xlabel("distance along transect (x, m)")
    ax_r.set_title("recharge (no ground truth; below the grey line = out of the ground)")

    out = run_dir / f"transect_stacked_{sweep}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(run_dir / f"transect_stacked_{sweep}.pdf", bbox_inches="tight")
    print(f"Saved {out}")
    return fig


if __name__ == "__main__":
    main(RUN_DIR)
    plt.show()

# %%
