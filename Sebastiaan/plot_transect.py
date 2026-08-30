"""
Transect plotter (conceptual).

Loads the truth + per-value fitted head fields written by transect_runner.py,
slices a horizontal transect at mid-domain, and plots:
    bold black line = IBRAHYM truth
    coloured lines  = Respighi fit at each swept value (kD or reg)
along the transect. One figure; the swept parameter is auto-detected from the
saved files. No re-solving -- pure read, so iterate freely.
"""
#%%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RUN_DIR = Path(r"C:\Users\sebas\Documents\Thesis Interpolating GW Levels\respighi-mastercopy\SavedData\transect_reg_20260826_2106")            # folder with truth.nc + head_*.nc, or its parent
TRANSECT_FRAC = 0.5                        # 0.5 = mid-domain; 0..1 across the y-range

_PAT = re.compile(r"head_([A-Za-z]+)(\d+)\.nc$")

#%%
def resolve(root: Path) -> Path:
    root = Path(root)
    if (root / "truth.nc").exists():
        return root
    subs = sorted(root.glob("transect_*"))
    if not subs:
        raise FileNotFoundError(f"No truth.nc or transect_* under {root.resolve()}")
    return subs[-1]


def transect_line(da2d, frac):
    """Head along a horizontal line at y = frac of the way through the y-range."""
    y = da2d["y"].values
    y_pick = y.min() + frac * (y.max() - y.min())
    line = da2d.sel(y=y_pick, method="nearest")
    return line["x"].values, line.values, float(line["y"].values)


def main(run_dir: Path):
    run_dir = resolve(run_dir)
    print(f"Reading from: {run_dir.resolve()}")

    truth = xr.open_dataset(run_dir / "truth.nc")["head"]
    xt, ht, y_used = transect_line(truth, TRANSECT_FRAC)

    files = sorted(run_dir.glob("head_*.nc"))
    if not files:
        raise FileNotFoundError("No head_*.nc files found.")
    sweep = _PAT.search(files[0].name).group(1)      # "kD" or "reg"
    sym = r"$\gamma$" if sweep == "reg" else sweep

    # sort by the numeric value
    def val(f):
        return int(_PAT.search(f.name).group(2))
    files = sorted(files, key=val)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(xt, ht, color="black", lw=3, alpha=0.8, label="IBRAHYM (truth)", zorder=10)

    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(files)))
    for f, c in zip(files, cmap):
        ds = xr.open_dataset(f)
        v = ds.attrs.get("value", val(f))
        x, h, _ = transect_line(ds["head"], TRANSECT_FRAC)
        ax.plot(x, h, color=c, lw=1.8, label=f"{sym} = {v:g}")

    unit = "m$^2$/day" if sweep == "kD" else ""
    ax.set_xlabel("distance along transect (x, m)")
    ax.set_ylabel("head (m)")
    ax.set_title(f"Fitted head along a mid-domain transect vs. {sym}"
                 + (f"  [{unit}]" if unit else "")
                 + f"\n(y = {y_used:.0f} m; truth in black)")
    ax.legend(title=sym, fontsize=9)

    out = run_dir / f"transect_{sweep}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(run_dir / f"transect_{sweep}.pdf", bbox_inches="tight")
    print(f"Saved {out}")
    return fig


if __name__ == "__main__":
    main(RUN_DIR)
    plt.show()

# %%
