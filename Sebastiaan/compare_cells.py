"""
Load two saved cells from the 3x3 sigma experiment and compare their recharge
fields side by side, annotated with each cell's head MAE.

Point RUN_DIR at the timestamped folder the experiment wrote, then pick the two
cells to compare (default: honest 0.20/0.20 vs over-smoothed 0.20/0.50). The
question this answers: does the lower-head-MAE cell pay for it with a flatter,
smeared recharge field? If so, that's the reg-weight degeneracy made visible.
"""
#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# ---- point this at your run folder ----
RUN_DIR = Path("../SavedData/sigma3x3_20260811_1248")   # refer to the dataset/subfolder you want to access
# If RUN_DIR is the parent, grab the most recent sigma3x3_* subfolder:
if not any(RUN_DIR.glob("run_*.nc")):
    subs = sorted(RUN_DIR.glob("sigma3x3_*"))
    if subs:
        RUN_DIR = subs[-1]
print(f"Reading from: {RUN_DIR.resolve()}")


def cell_name(sigma_ext, sigma_int, repeat=0):
    ext_cm = int(round(sigma_ext * 100))
    int_cm = int(round(sigma_int * 100))
    return f"run_ext{ext_cm:02d}cm_int{int_cm:02d}cm_rep{repeat}.nc"


def load_cell(sigma_ext, sigma_int, repeat=0):
    ds = xr.open_dataset(RUN_DIR / cell_name(sigma_ext, sigma_int, repeat))
    return ds


def compare(sigma_ext, sigma_int_a, sigma_int_b, repeat=0):
    """Side-by-side recharge for two sigma_int at the same sigma_ext."""
    a = load_cell(sigma_ext, sigma_int_a, repeat)
    b = load_cell(sigma_ext, sigma_int_b, repeat)
    ra, rb = a["recharge"], b["recharge"]

    # Shared colour scale so the two panels are honestly comparable.
    vmin = float(min(ra.min(), rb.min()))
    vmax = float(max(ra.max(), rb.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, r, s_int, ds in ((axes[0], ra, sigma_int_a, a),
                             (axes[1], rb, sigma_int_b, b)):
        im = r.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, add_colorbar=False)
        std = float(np.nanstd(r.values))
        ax.set_title(
            f"$\\sigma_{{ext}}$={sigma_ext:.2f}, $\\sigma_{{int}}$={s_int:.2f}\n"
            f"head MAE = {ds.attrs['mae_head']:.4f} m   |   "
            f"recharge std = {std:.2e}"
        )
        ax.set_aspect("equal")
    fig.colorbar(im, ax=axes, shrink=0.8, label="recharge [m/d]")
    fig.suptitle("Recharge: honest vs over-smoothed "
                 "(lower head MAE, but flatter field?)", fontsize=12)

    out = RUN_DIR / f"compare_recharge_ext{int(sigma_ext*100):02d}cm.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"  {sigma_int_a:.2f}: head MAE {a.attrs['mae_head']:.4f}, "
          f"recharge std {float(np.nanstd(ra.values)):.3e}")
    print(f"  {sigma_int_b:.2f}: head MAE {b.attrs['mae_head']:.4f}, "
          f"recharge std {float(np.nanstd(rb.values)):.3e}")
    return fig


if __name__ == "__main__":
    # Honest vs over-smoothed at the middle noise level.
    compare(sigma_ext=0.20, sigma_int_a=0.20, sigma_int_b=0.50)
    plt.show()

# %%
