#%%
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RUN_DIR = Path("../SavedData")     # folder with head_*.nc, or its parent
TRANSECT_FRAC = 0.5

_PAT = re.compile(r"head_([A-Za-z]+)(\d+)\.nc$")

#%%
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

def main(run_dir):
    run_dir = resolve(run_dir)
    print(f"Reading from: {run_dir.resolve()}")

    files = sorted(run_dir.glob("head_*.nc"))
    if not files:
        raise FileNotFoundError("No head_*.nc files found.")
    sweep = _PAT.search(files[0].name).group(1)
    files = sorted(files, key=lambda f: int(_PAT.search(f.name).group(2)))

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(files)))
    y_used = None
    for f, c in zip(files, cmap):
        ds = xr.open_dataset(f)
        if "recharge" not in ds:
            raise KeyError(f"{f.name} has no 'recharge' — re-run the runner with the recharge save added.")
        v = ds.attrs.get("value", int(_PAT.search(f.name).group(2)))
        x, r, y_used = transect_line(ds["recharge"], TRANSECT_FRAC)
        ax.plot(x, r, color=c, lw=1.8, label=f"{sweep} = {v:g}")

    ax.axhline(0, color="gray", lw=0.8, zorder=0)
    unit = "m$^2$/day" if sweep == "kD" else ""
    ax.set_xlabel("distance along transect (x, m)")
    ax.set_ylabel("fitted recharge (m/day)")
    ax.set_title(f"Fitted recharge along a mid-domain transect vs. {sweep}"
                 + (f"  [{unit}]" if unit else "")
                 + f"\n(y = {y_used:.0f} m; no recharge ground truth)")
    ax.legend(title=sweep, fontsize=9)

    out = run_dir / f"transect_recharge_{sweep}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(run_dir / f"transect_recharge_{sweep}.pdf", bbox_inches="tight")
    print(f"Saved {out}")
    return fig

if __name__ == "__main__":
    main(RUN_DIR)
    plt.show()