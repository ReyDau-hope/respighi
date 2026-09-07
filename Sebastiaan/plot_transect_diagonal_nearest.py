"""
Diagonal transect plotter (bottom-left -> top-right, nearest-cell sampling).

Same as the horizontal transect plotter, but slices a diagonal line across the
domain from (x_min, y_min) to (x_max, y_max). Nearest-cell sampling along the
line (simple, slightly blocky). x-axis is distance ALONG the diagonal (m).

Loads truth + per-value head fields, plots truth (black) + one curve per swept
value. Auto-detects kD / reg / sigma from filenames. Pure read.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RUN_DIR = Path(r"C:\Users\sebas\Documents\1Thesis\respighi\SavedData\transect_kD_XXXX")  # <- set
SELECT_VALUES = None               # None = all curves; else e.g. [250, 1000, 4000]

_PAT = re.compile(r"head_([A-Za-z]+)(\d+)\.nc$")


def resolve(root: Path) -> Path:
    root = Path(root)
    if (root / "truth.nc").exists():
        return root
    subs = sorted(root.glob("transect_*"))
    if not subs:
        raise FileNotFoundError(f"No truth.nc or transect_* under {root.resolve()}")
    return subs[-1]


def n_samples(da2d):
    """Roughly one sample per cell along the diagonal."""
    x = da2d["x"].values
    dx = abs(np.median(np.diff(np.sort(x))))
    x0, x1 = float(x.min()), float(x.max())
    y = da2d["y"].values
    y0, y1 = float(y.min()), float(y.max())
    length = np.hypot(x1 - x0, y1 - y0)
    return max(int(length / dx), 10)


def diagonal_line(da2d, n):
    """Nearest-cell sampling along BL->TR diagonal. Returns (distance, values)."""
    x = da2d["x"].values
    y = da2d["y"].values
    x0, x1 = float(x.min()), float(x.max())     # bottom-left -> top-right
    y0, y1 = float(y.min()), float(y.max())
    xline = np.linspace(x0, x1, n)
    yline = np.linspace(y0, y1, n)
    dist = np.hypot(xline - x0, yline - y0)      # metres along the diagonal
    sampled = da2d.sel(
        x=xr.DataArray(xline, dims="s"),
        y=xr.DataArray(yline, dims="s"),
        method="nearest",
    )
    return dist, sampled.values


def main(run_dir: Path):
    run_dir = resolve(run_dir)
    print(f"Reading from: {run_dir.resolve()}")

    truth = xr.open_dataset(run_dir / "truth.nc")["head"]
    n = n_samples(truth)
    dist, ht = diagonal_line(truth, n)

    files = sorted(run_dir.glob("head_*.nc"))
    if not files:
        raise FileNotFoundError("No head_*.nc files found.")
    sweep = _PAT.search(files[0].name).group(1)
    sym = (r"$\gamma$" if sweep == "reg" else
           r"$\sigma$" if sweep == "sigma" else sweep)

    def val(f):
        return int(_PAT.search(f.name).group(2))
    files = sorted(files, key=val)

    if SELECT_VALUES is not None:
        wanted = {int(round(v)) for v in SELECT_VALUES}
        files = [f for f in files if val(f) in wanted]
        if not files:
            raise ValueError(f"None of SELECT_VALUES {SELECT_VALUES} matched.")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(dist, ht, color="black", lw=2.5, alpha=0.85,
            label="IBRAHYM (truth)", zorder=10)

    colors = plt.cm.tab10(range(len(files)))
    for f, c in zip(files, colors):
        ds = xr.open_dataset(f)
        v = ds.attrs.get("value", val(f))
        _, h = diagonal_line(ds["head"], n)
        ax.plot(dist, h, color=c, lw=1.8, label=f"{sym} = {v:g}")

    unit = "m$^2$/day" if sweep == "kD" else ""
    ax.set_xlabel("distance along diagonal transect (m)")
    ax.set_ylabel("head (m)")
    ax.set_title(f"Fitted head along a diagonal transect (BL$\\rightarrow$TR) vs. {sym}"
                 + (f"  [{unit}]" if unit else "")
                 + "\n(truth in black)")
    ax.legend(title=sym, fontsize=9)

    out = run_dir / f"transect_diagonal_{sweep}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(run_dir / f"transect_diagonal_{sweep}.pdf", bbox_inches="tight")
    print(f"Saved {out}")
    return fig


if __name__ == "__main__":
    main(RUN_DIR)
    plt.show()
