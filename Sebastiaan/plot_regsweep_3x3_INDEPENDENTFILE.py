"""
Standalone plotter for the regularization sweep.

Reads whatever reg######/ subfolders exist under RUN_DIR, rebuilds the
(reg_weight, sigma_ext, sigma_int) head-MAE cube from the saved run_*.nc cells,
and draws the convergence curve + the per-reg 3x3 grid. No solving, no dependency
on regsweep_summary.nc -- so it works no matter how the cells got there (single
run, resumed run, or two runs merged into one folder by copying reg###### dirs).

Usage: set RUN_DIR to the folder holding the reg###### subdirs, then run.
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RUN_DIR = Path("../SavedData/sigma_regsweep_20260813_1251")   # folder containing reg000500/, reg001000/, ...

_REG = re.compile(r"reg(\d+)$")
_CELL = re.compile(r"run_ext(\d+)cm_int(\d+)cm_rep(\d+)\.nc$")


def resolve(root: Path) -> Path:
    root = Path(root)
    if any(root.glob("reg*")):
        return root
    subs = sorted(root.glob("sigma_regsweep_*")) or sorted(root.glob("sigma3x3_*"))
    if not subs:
        raise FileNotFoundError(f"No reg*/ subfolders under {root.resolve()}")
    return subs[-1]


def rebuild_cube(run_dir: Path):
    """Return (reg_weights, ext, int, MAE[reg,ext,int]) averaged over repeats,
    read from every reg######/run_*.nc under run_dir."""
    acc = defaultdict(list)                       # (w, e, s) -> [mae per rep]
    regs, exts, ints = set(), set(), set()
    for reg_dir in sorted(run_dir.glob("reg*")):
        m = _REG.search(reg_dir.name)
        if not m:
            continue
        w = float(int(m.group(1)))
        for f in sorted(reg_dir.glob("run_*.nc")):
            cm = _CELL.search(f.name)
            if not cm:
                continue
            e, s = int(cm.group(1)) / 100, int(cm.group(2)) / 100
            regs.add(w); exts.add(e); ints.add(s)
            with xr.open_dataset(f) as ds:
                acc[(w, e, s)].append(float(ds.attrs.get("mae_head", np.nan)))
    regs, exts, ints = sorted(regs), sorted(exts), sorted(ints)
    M = np.full((len(regs), len(exts), len(ints)), np.nan)
    for a, w in enumerate(regs):
        for i, e in enumerate(exts):
            for j, s in enumerate(ints):
                if acc[(w, e, s)]:
                    M[a, i, j] = np.nanmean(acc[(w, e, s)])
    return regs, exts, ints, M


def plot_grid(regs, ext, intv, M, out_dir):
    ncol = min(len(regs), 3)
    nrow = int(np.ceil(len(regs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow),
                             constrained_layout=True, squeeze=False)
    vmin, vmax = np.nanmin(M), np.nanmax(M)
    for k, w in enumerate(regs):
        ax = axes[k // ncol][k % ncol]
        m = M[k]
        im = ax.imshow(m, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(intv)), [f"{v:.2f}" for v in intv])
        ax.set_yticks(range(len(ext)), [f"{v:.2f}" for v in ext])
        ax.set_title(f"reg = {w:g}")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.3g}", ha="center", va="center",
                            fontsize=7,
                            color="white" if im.norm(m[i, j]) < 0.6 else "black")
            if np.any(np.isfinite(m[i])):
                aj = int(np.nanargmin(m[i]))
                ax.plot(aj, i, "o", mfc="none", mec="red", ms=16, mew=2)
        for t in range(min(m.shape)):
            ax.add_patch(plt.Rectangle((t - 0.5, t - 0.5), 1, 1, fill=False,
                                       edgecolor="white", lw=1, ls="--"))
        ax.set_xlabel(r"$\sigma_{int}$"); ax.set_ylabel(r"$\sigma_{ext}$")
    for k in range(len(regs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(r"row-min (red $\circ$) vs honest diagonal (dashed)", fontsize=12)
    out = Path(out_dir) / "regsweep_grid_merged.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return fig


def plot_convergence(regs, ext, intv, M, out_dir):
    regs = np.asarray(regs, float)
    diag, rmin = np.full(len(regs), np.nan), np.full(len(regs), np.nan)
    for k in range(len(regs)):
        d = [M[k, i, int(np.argmin(np.abs(np.array(intv) - e)))]
             for i, e in enumerate(ext)]
        diag[k] = np.nanmean(d)
        rmin[k] = np.nanmean([np.nanmin(M[k, i]) for i in range(len(ext))])
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.plot(regs, diag, "-o", label="honest (diagonal) mean MAE")
    ax.plot(regs, rmin, "-s", label="best-case (row-min) mean MAE")
    ax.set_xscale("log")
    ax.set_xlabel("regularization weight")
    ax.set_ylabel("mean head MAE [m]")
    ax.set_title("Where the curves meet, honesty costs nothing")
    ax.legend()
    out = Path(out_dir) / "regsweep_convergence_merged.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return fig


if __name__ == "__main__":
    run_dir = resolve(RUN_DIR)
    print(f"Reading from: {run_dir.resolve()}")
    regs, ext, intv, M = rebuild_cube(run_dir)
    print(f"Found reg weights: {[f'{w:g}' for w in regs]}")
    plot_grid(regs, ext, intv, M, run_dir)
    plot_convergence(regs, ext, intv, M, run_dir)
    plt.show()