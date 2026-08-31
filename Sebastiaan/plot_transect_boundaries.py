"""
Boundary-condition overlay for the transect figures (discrete features only).

Marks where the transect crosses a DISCRETE boundary feature -- drains, rivers,
largerivers (tiledrainage optional). Overland flow is deliberately excluded: it
is an areal Drainage boundary active almost everywhere, so it tells you nothing
about *where* a feature is. Recharge is a source term, not a head boundary, so
it's not here either.

Two figures:
  (1) dual-axis head/flux plot with boundary TICKS along the bottom, and
  (2) standalone net-flux-vs-boundaries figure.

Point RUN_DIR at the transect folder AND set BASE + the window to match the
runner. "Active" = conductance > CONDUCTANCE_MIN.
"""
#%%
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# --- transect data ---
RUN_DIR = Path(r"C:\Users\sebas\Documents\1Thesis\respighi\SavedData\transect_kD_20260826_2051")     # transect_* folder, or its parent
TRANSECT_FRAC = 0.5
FEATURE_VALUES = None              # None = lowest & highest swept value

# --- IBRAHYM boundary data (MUST match the runner) ---
BASE = r"C:\Users\sebas\Documents\1Thesis\case\ibrahym\ibrahym-"  # <-- RE-POINT to your data path
SCENARIO = ""
XMIN, XMAX = 185_000.0, 205_000.0
YMIN, YMAX = 350_000.0, 370_000.0

INCLUDE_TILEDRAIN = False          # toggle tiledrainage in/out of the mask
CONDUCTANCE_MIN = 0.0              # a cell counts as a boundary if conductance > this

_PAT = re.compile(r"head_([A-Za-z]+)(\d+)\.nc$")
C_HEAD, C_RECH = "tab:blue", "tab:red"


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


def boundary_mask_along_transect(y_pick, verbose=True):
    """Combined mask over DISCRETE boundaries (drains, rivers, largerivers,
    optionally tiledrainage) sampled along the transect. Returns (x, bool)."""
    def slice_ds(ds):
        return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))

    specs = [
        ("drains",      f"{BASE}drains-100m{SCENARIO}.nc",      "conductance"),
        ("rivers",      f"{BASE}rivers-100m{SCENARIO}.nc",      "conductance"),
        ("largerivers", f"{BASE}largerivers-100m{SCENARIO}.nc", "conductance"),
    ]
    if INCLUDE_TILEDRAIN:
        specs.append(("tiledrain", f"{BASE}tiledrainage-100m{SCENARIO}.nc", "conductance"))

    combined, xref = None, None
    for name, path, field in specs:
        try:
            ds = slice_ds(xr.open_dataset(path))
        except FileNotFoundError:
            warnings.warn(f"boundary file missing, skipped: {path}")
            continue
        da = ds[field]
        if "time" in da.dims:
            da = da.isel(time=0)
        active2d = np.isfinite(da) & (da > CONDUCTANCE_MIN)
        line = active2d.sel(y=y_pick, method="nearest")
        xref = line["x"].values
        vals = np.asarray(line.values, dtype=bool)
        if verbose:
            print(f"    {name:12s}: {int(vals.sum()):3d} / {vals.size} cells")
        combined = vals if combined is None else (combined | vals)
    if combined is None:
        raise FileNotFoundError("No boundary files found -- check BASE.")
    return xref, combined


def mark_boundaries(ax, x, active, color="black", label="boundary (drain/river)"):
    """Tick markers along the bottom axis at each boundary cell."""
    xb = np.asarray(x)[np.asarray(active, dtype=bool)]
    if xb.size == 0:
        return
    ax.plot(xb, np.full_like(xb, 0.0), marker="|", ms=14, mew=1.5, ls="none",
            color=color, transform=ax.get_xaxis_transform(), clip_on=False,
            label=label, zorder=6)


def main(run_dir):
    run_dir = resolve(run_dir)
    print(f"Reading from: {run_dir.resolve()}")
    sweep, all_vals = sweep_and_values(run_dir)
    feats = FEATURE_VALUES if FEATURE_VALUES else [all_vals[0], all_vals[-1]]

    truth = xr.open_dataset(run_dir / "truth.nc")["head"]
    xt, ht, y_used = transect_line(truth, TRANSECT_FRAC)
    print(f"Transect at y = {y_used:.0f} m; per-boundary coverage:")
    xb, active = boundary_mask_along_transect(y_used)
    print(f"    combined    : {int(active.sum())} / {active.size} cells cross a boundary")

    # ---------- Figure 1: dual-axis WITH boundary ticks ----------
    fig1, axes = plt.subplots(1, len(feats), figsize=(6.5 * len(feats), 4.8),
                              constrained_layout=True, squeeze=False)
    for ax, v in zip(axes[0], feats):
        ds = xr.open_dataset(find_file(run_dir, sweep, v))
        x, h, _ = transect_line(ds["head"], TRANSECT_FRAC)
        _, r, _ = transect_line(ds["recharge"], TRANSECT_FRAC)
        ax.plot(xt, ht, color="black", lw=2.5, label="truth", zorder=5)
        ax.plot(x, h, color=C_HEAD, lw=1.8, label="fitted head")
        mark_boundaries(ax, xb, active)
        ax.set_xlabel("distance along transect (x, m)")
        ax.set_ylabel("head (m)", color=C_HEAD)
        ax.tick_params(axis="y", labelcolor=C_HEAD)
        ax.set_title(f"{sweep} = {v:g}")
        ax2 = ax.twinx()
        ax2.plot(x, r, color=C_RECH, lw=1.6, alpha=0.9)
        ax2.axhline(0, color=C_RECH, ls=":", lw=1.0)
        rmax = float(np.nanmax(np.abs(r)))
        ax2.set_ylim(-1.1 * rmax, 1.1 * rmax)
        ax2.set_ylabel("net vertical flux (m/day)", color=C_RECH)
        ax2.tick_params(axis="y", labelcolor=C_RECH)
        ax.legend(loc="upper center", fontsize=8)
    fig1.suptitle(f"Head (blue) & net flux (red); ticks = drain/river crossings  "
                  f"(y = {y_used:.0f} m)", fontsize=12)
    out1 = run_dir / f"transect_dualaxis_boundaries_{sweep}.png"
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    fig1.savefig(run_dir / f"transect_dualaxis_boundaries_{sweep}.pdf", bbox_inches="tight")
    print(f"Saved {out1}")

    # ---------- Figure 2: standalone net flux vs boundary ticks ----------
    fig2, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(feats)))
    for v, c in zip(feats, colors):
        ds = xr.open_dataset(find_file(run_dir, sweep, v))
        x, r, _ = transect_line(ds["recharge"], TRANSECT_FRAC)
        ax.plot(x, r, color=c, lw=1.8, label=f"{sweep} = {v:g}")
    ax.axhline(0, color="gray", lw=1.0, zorder=0)
    mark_boundaries(ax, xb, active)
    ax.set_xlabel("distance along transect (x, m)")
    ax.set_ylabel("net vertical flux (m/day)")
    ax.set_title(f"Net vertical flux vs. drain/river crossings  (y = {y_used:.0f} m)")
    ax.legend(fontsize=9)
    out2 = run_dir / f"transect_flux_boundaries_{sweep}.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    fig2.savefig(run_dir / f"transect_flux_boundaries_{sweep}.pdf", bbox_inches="tight")
    print(f"Saved {out2}")
    return fig1, fig2


if __name__ == "__main__":
    main(RUN_DIR)
    plt.show()

# %%
