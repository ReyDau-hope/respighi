"""
kD sweep (single conductance scenario, noise-free), current API.

Sweeps kD at a fixed gamma, records head MAE vs the IBRAHYM truth, and CAPTURES
the convergence flag for every solve (so you never trust non-converged numbers
again). Saves the curve + convergence info to a netCDF in SavedData, plus the
log-log plot. relax=1.0 throughout.

Standalone -- run start to finish. RE-POINT `BASE` if needed.
"""
#%%
from __future__ import annotations

import gc
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xugrid as xu

import respighi as rsp

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEED = 12345
GAMMA = 10.0                    # regularization weight
RECHARGE = 0.001
N_PIEZOMETERS = 200
KD_VALUES = np.logspace(np.log10(200), np.log10(10000), 40)   # floor at 200 to catch low minima
SCENARIO = ""                     # "", "-cond0.5", "-cond2", "-cond3"

XMIN, XMAX = 185_000.0, 205_000.0
YMIN, YMAX = 350_000.0, 370_000.0
BASE = r"C:\Users\sebas\Documents\1Thesis\case\ibrahym\ibrahym-"   # <-- RE-POINT

SAVE_ROOT = Path("../SavedData")
scenario_label = SCENARIO.lstrip("-") if SCENARIO else "original"
RUN_DIR = SAVE_ROOT / f"kD_sweep_{scenario_label}_{datetime.now():%Y%m%d_%H%M}"


def slice_dataset(ds):
    return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))


def run_experiment():
    # ---- setup (once) ----
    head = xr.open_dataset(f"{BASE}head-l1-100m.nc")["head"]
    modelhead = slice_dataset(head.isel(time=-1))
    drain_ds        = slice_dataset(xr.open_dataset(f"{BASE}drains-100m{SCENARIO}.nc"))
    river_ds        = slice_dataset(xr.open_dataset(f"{BASE}rivers-100m{SCENARIO}.nc")).isel(time=0)
    large_river_ds  = slice_dataset(xr.open_dataset(f"{BASE}largerivers-100m{SCENARIO}.nc"))
    tiledrain_ds    = slice_dataset(xr.open_dataset(f"{BASE}tiledrainage-100m{SCENARIO}.nc"))
    overlandflow_ds = slice_dataset(xr.open_dataset(f"{BASE}overlandflow-100m.nc"))
    subsoil         = slice_dataset(xr.open_dataset(f"{BASE}subsoil-100m.nc"))
    hfb_gdf         = gpd.read_file(f"{BASE.replace('ibrahym-', '')}hfb-12.gpkg")

    kh_template = subsoil["kh"].isel(layer=0, drop=True)
    hfb = rsp.HorizontalFlowBarrier.from_geodataframe(
        layer=0, barriers=hfb_gdf,
        template=xr.full_like(kh_template, 1.0), max_snap_distance=10.0,
    )
    boundaries = [
        rsp.River.from_dataset(river_ds),
        rsp.River.from_dataset(large_river_ds),
        rsp.Drainage.from_dataset(drain_ds),
        rsp.Drainage.from_dataset(tiledrain_ds),
        rsp.Drainage.from_dataset(overlandflow_ds, constant_conductance=500.0),
    ]

    rng = np.random.default_rng(seed=SEED)
    px = XMIN + (XMAX - XMIN) * rng.random(N_PIEZOMETERS)
    py = YMIN + (YMAX - YMIN) * rng.random(N_PIEZOMETERS)
    grid = xu.Ugrid2d.from_structured(modelhead)
    headvalues = modelhead.sel(
        x=xr.DataArray(px), y=xr.DataArray(py), method="nearest"
    ).to_numpy()
    target = rsp.CellSampling(px, py, headvalues, grid)      # noise-free

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    n = len(KD_VALUES)
    mae = np.full(n, np.nan)
    converged = np.zeros(n, dtype=int)
    iters = np.zeros(n, dtype=int)
    print(f"Sweeping {n} kD values at gamma={GAMMA:g}, scenario={scenario_label}\n")

    for ki, kD in enumerate(KD_VALUES):
        transmissivity = xr.full_like(kh_template, float(kD))
        recharge = rsp.Recharge(rate=xr.full_like(transmissivity, RECHARGE).to_numpy())
        gwf = rsp.GroundwaterModel(
            area=100.0 * 100.0, initial=modelhead, recharge=recharge,
            head_boundaries=boundaries, transmissivity=transmissivity,
            horizontal_flow_barriers=[hfb], xclose=1e-6, maxiter=50,
        )
        gwf.formulate()
        gwf.nonlinear_solve()
        inverse = rsp.InverseProblem(
            gwf, target, regularization=rsp.UnscaledMinimumCurvature(GAMMA),
            maxiter=100, relax=1.0,          # <-- 1.0, NOT 0.0
        )
        inverse.formulate()
        conv, nit = inverse.nonlinear_solve()      # capture the flag
        err = inverse.head.isel(layer=0) - modelhead
        mae[ki] = float(abs(err).mean())
        converged[ki] = int(conv)
        iters[ki] = int(nit)

        flag = "" if conv else "  <-- NOT CONVERGED"
        print(f"  kD={kD:7.0f}  MAE={mae[ki]:.4f}  iters={nit}{flag}")

        for obj in (inverse, gwf):
            ls = getattr(obj, "linearsolver", None)
            if ls is not None and hasattr(ls, "free_memory"):
                ls.free_memory()
        del inverse, gwf, err, transmissivity, recharge
        gc.collect()

    # ---- save ----
    result = xr.Dataset(
        {"mae": ("kD", mae),
         "converged": ("kD", converged),
         "iterations": ("kD", iters)},
        coords={"kD": KD_VALUES},
        attrs={"gamma": float(GAMMA), "recharge": float(RECHARGE), "seed": SEED,
               "scenario": scenario_label, "n_piezometers": N_PIEZOMETERS,
               "noise": "none"},
    )
    result.to_netcdf(RUN_DIR / "kD_sweep.nc")

    # ---- report ----
    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(KD_VALUES, mae, marker=".", color="tab:blue", label="all solves", zorder=1)
    bad = converged == 0
    if bad.any():
        ax.plot(KD_VALUES[bad], mae[bad], "x", color="red", ms=9,
                label="not converged", zorder=3)

    # scale the y-axis to the CONVERGED points, so one bad spike doesn't
    # flatten the real signal. Non-converged points stay marked but may
    # clip above the top (that's fine -- you can still see failures exist).
    good = converged == 1
    if good.any():
        lo = float(np.nanmin(mae[good]))
        hi = float(np.nanmax(mae[good]))
        pad = 0.1 * (hi - lo) if hi > lo else 0.05
        ax.set_ylim(lo - pad, hi + pad)
        n_clipped = int(np.sum(mae[bad] > hi + pad)) if bad.any() else 0
        if n_clipped:
            ax.text(0.98, 0.98,
                    f"{n_clipped} non-converged point(s) clipped above",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="red")

    ax.set_xscale("log")
    ax.set_xlabel(r"kD (m$^2$/day)")
    ax.set_ylabel("mean absolute error (m)")
    ax.set_title(f"Error vs transmissivity  ($\\gamma$ = {GAMMA:g}, {scenario_label})")
    ax.legend()
    fig.savefig(RUN_DIR / "kD_sweep.png", dpi=200, bbox_inches="tight")
    fig.savefig(RUN_DIR / "kD_sweep.pdf", bbox_inches="tight")
    print(f"\nSaved to {RUN_DIR.resolve()}")
    return result


if __name__ == "__main__":
    run_experiment()
    plt.show()

