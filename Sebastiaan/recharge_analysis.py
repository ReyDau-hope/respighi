"""
Recharge-side analysis of the 3x3 sigma experiment.

Two figures, both built from the saved run_*.nc cells (no re-solving):

  1. std_matrix()   -- head-MAE and recharge-std 3x3 matrices side by side,
     averaged over repeats. Both fall left-to-right (rising sigma_int): the
     lowest-head-error cells are ALSO the most-flattened recharge fields. That
     coincidence IS the trade-off -- best fit == least structure.

  2. histograms()   -- overlaid recharge value distributions for two cells
     (default: honest 0.20/0.20 vs over-smoothed 0.20/0.50). The over-smoothed
     field's distribution is narrower, its mass collapsing toward the mean --
     the blur from the map, shown as a distribution. (This is the "histograms
     of spatial recharge distribution" from the meeting notes.)

Point RUN_DIR at the experiment folder (or its parent -- the latest sigma3x3_*
subfolder is picked automatically). Metrics are read from each file's attrs and
recharge fields, so this is self-contained; it does not need mae_matrix_3x3.nc.
"""
#%%
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# ---- point this at your run folder, or its parent ----
RUN_DIR = Path("../SavedData/sigma3x3_20260811_1248")

_PAT = re.compile(r"run_ext(\d+)cm_int(\d+)cm_rep(\d+)\.nc$")


def resolve_run_dir(root: Path) -> Path:
    root = Path(root)
    if any(root.glob("run_*.nc")):
        return root
    subs = sorted(root.glob("sigma3x3_*"))
    if not subs:
        raise FileNotFoundError(f"No run_*.nc or sigma3x3_* under {root.resolve()}")
    return subs[-1]


def scan(run_dir: Path):
    """Return (ext_values, int_values, head_MAE_matrix, recharge_std_matrix),
    each matrix averaged over repeats. Read straight from the saved files."""
    mae = defaultdict(list)
    std = defaultdict(list)
    exts, ints = set(), set()
    for f in sorted(run_dir.glob("run_*.nc")):
        m = _PAT.search(f.name)
        if not m:
            continue
        e, s = int(m.group(1)) / 100, int(m.group(2)) / 100
        exts.add(e)
        ints.add(s)
        with xr.open_dataset(f) as ds:
            mae[(e, s)].append(float(ds.attrs.get("mae_head", np.nan)))
            std[(e, s)].append(float(np.nanstd(ds["recharge"].values)))
    exts, ints = sorted(exts), sorted(ints)
    MAE = np.full((len(exts), len(ints)), np.nan)
    STD = np.full((len(exts), len(ints)), np.nan)
    for i, e in enumerate(exts):
        for j, s in enumerate(ints):
            if mae[(e, s)]:
                MAE[i, j] = np.nanmean(mae[(e, s)])
            if std[(e, s)]:
                STD[i, j] = np.nanmean(std[(e, s)])
    return exts, ints, MAE, STD


def _annotated_heatmap(ax, M, ext, intv, title, fmt):
    im = ax.imshow(M, origin="upper", cmap="viridis")
    ax.set_xticks(range(len(intv)), [f"{v:.2f}" for v in intv])
    ax.set_yticks(range(len(ext)), [f"{v:.2f}" for v in ext])
    ax.set_xlabel(r"$\sigma_{\mathrm{int}}$ (told) [m]")
    ax.set_ylabel(r"$\sigma_{\mathrm{ext}}$ (injected) [m]")
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                        color="white" if im.norm(v) < 0.6 else "black")
    for k in range(min(M.shape)):  # outline the honest diagonal
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="red", lw=2))
    return im


def std_matrix(run_dir: Path):
    ext, intv, MAE, STD = scan(run_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    im0 = _annotated_heatmap(
        axes[0], MAE, ext, intv,
        "head MAE  [m]\n(lower = better head fit)", "{:.3g}")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = _annotated_heatmap(
        axes[1], STD, ext, intv,
        "recharge std  [m/d]\n(lower = flatter, less structure)", "{:.2e}")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.suptitle("Both fall left-to-right: the best-fitting cells are the "
                 "most-flattened recharge fields", fontsize=12)

    out = run_dir / "matrix_headMAE_vs_rechargeStd.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print("\nhead MAE (rows=ext, cols=int):\n", np.array_str(MAE, precision=4))
    print("\nrecharge std:\n", np.array_str(STD, precision=4))
    return fig


def _pooled_recharge(run_dir: Path, sigma_ext, sigma_int):
    ext_cm, int_cm = int(round(sigma_ext * 100)), int(round(sigma_int * 100))
    vals = []
    for f in sorted(run_dir.glob(f"run_ext{ext_cm:02d}cm_int{int_cm:02d}cm_rep*.nc")):
        with xr.open_dataset(f) as ds:
            v = ds["recharge"].values.ravel()
            vals.append(v[np.isfinite(v)])
    if not vals:
        raise FileNotFoundError(f"No cell for ext={sigma_ext}, int={sigma_int}")
    return np.concatenate(vals)


def histograms(run_dir: Path, sigma_ext=0.20, sigma_int_a=0.20, sigma_int_b=0.50):
    va = _pooled_recharge(run_dir, sigma_ext, sigma_int_a)
    vb = _pooled_recharge(run_dir, sigma_ext, sigma_int_b)

    both = np.concatenate([va, vb])
    lo, hi = np.percentile(both, [0.5, 99.5])   # trim outliers for display only
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.hist(va, bins=bins, density=True, alpha=0.55,
            label=f"$\\sigma_{{int}}$={sigma_int_a:.2f}  (honest, std {va.std():.2e})")
    ax.hist(vb, bins=bins, density=True, alpha=0.55,
            label=f"$\\sigma_{{int}}$={sigma_int_b:.2f}  (over-smoothed, std {vb.std():.2e})")
    ax.axvline(va.mean(), color="C0", ls="--", lw=1)
    ax.axvline(vb.mean(), color="C1", ls="--", lw=1)
    ax.set_xlabel("recharge [m/d]")
    ax.set_ylabel("density")
    ax.set_title(f"Recharge distribution at $\\sigma_{{ext}}$={sigma_ext:.2f}\n"
                 "over-smoothed field is narrower -- structure collapsing to the mean")
    ax.legend()

    out = run_dir / f"recharge_hist_ext{int(sigma_ext*100):02d}cm.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"  int={sigma_int_a:.2f}: std {va.std():.3e}, range {va.ptp():.3e}")
    print(f"  int={sigma_int_b:.2f}: std {vb.std():.3e}, range {vb.ptp():.3e}")
    return fig


if __name__ == "__main__":
    run_dir = resolve_run_dir(RUN_DIR)
    print(f"Reading from: {run_dir.resolve()}\n")
    std_matrix(run_dir)
    histograms(run_dir, sigma_ext=0.20, sigma_int_a=0.20, sigma_int_b=0.50)
    plt.show()
