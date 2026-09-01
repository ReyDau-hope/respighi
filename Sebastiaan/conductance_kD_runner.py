"""
Conductance-variant kD sweep (noise-free, current API).

For each conductance scenario (original, x0.5, x2, x3), sweeps kD and records the
head MAE against the IBRAHYM truth. Shows two things at once: the kD optimum, and
how that optimum SHIFTS with drainage conductance -- the kD<->conductance
collinearity (the c<->kD trade-off from the analytical h-bar expression, shown
empirically).

Noise-free (one clean solve per kD), gamma fixed, relax=1.0. Saves the curves to
one netCDF; companion plotter: plot_conductance_kD.py. RE-POINT `BASE`.
"""

from __future__ import annotations

import gc
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu

import respighi as rsp

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEED = 12345
GAMMA = 4000.0                    # regularization weight (set as you like)
RECHARGE = 0.001
N_PIEZOMETERS = 200
KD_VALUES = np.logspace(np.log10(1000), np.log10(10000), 40)

XMIN, XMAX = 185_000.0, 205_000.0
YMIN, YMAX = 350_000.0, 370_000.0
BASE = "../case/ibrahym/ibrahym-"     # <-- RE-POINT to your data path

VARIANTS = {                       # label -> conductance-file suffix
    "Original":  "",
    "cond x0.5": "-cond0.5",
    "cond x2":   "-cond2",
    "cond x3":   "-cond3",
}

SAVE_ROOT = Path("../SavedData")
RUN_DIR = SAVE_ROOT / f"conductance_kD_{datetime.now():%Y%m%d_%H%M}"


def slice_dataset(ds):
    return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))


def run_experiment():
    # ---- shared setup, once (truth, subsoil, HFB, overlandflow, target) ----
    head = xr.open_dataset(f"{BASE}head-l1-100m.nc")["head"]
    modelhead = slice_dataset(head.isel(time=-1))
    subsoil = slice_dataset(xr.open_dataset(f"{BASE}subsoil-100m.nc"))
    overlandflow_ds = slice_dataset(xr.open_dataset(f"{BASE}overlandflow-100m.nc"))
    hfb_gdf = gpd.read_file(f"{BASE.replace('ibrahym-', '')}hfb-12.gpkg")

    kh_template = subsoil["kh"].isel(layer=0, drop=True)
    hfb = rsp.HorizontalFlowBarrier.from_geodataframe(
        layer=0, barriers=hfb_gdf,
        template=xr.full_like(kh_template, 1.0), max_snap_distance=10.0,
    )
    overlandflow = rsp.Drainage.from_dataset(overlandflow_ds, constant_conductance=500.0)

    rng = np.random.default_rng(seed=SEED)
    px = XMIN + (XMAX - XMIN) * rng.random(N_PIEZOMETERS)
    py = YMIN + (YMAX - YMIN) * rng.random(N_PIEZOMETERS)
    grid = xu.Ugrid2d.from_structured(modelhead)
    headvalues = modelhead.sel(
        x=xr.DataArray(px), y=xr.DataArray(py), method="nearest"
    ).to_numpy()
    target = rsp.CellSampling(px, py, headvalues, grid)      # noise-free, shared

    def build_boundaries(suffix):
        river_ds = slice_dataset(xr.open_dataset(f"{BASE}rivers-100m{suffix}.nc")).isel(time=0)
        large_river_ds = slice_dataset(xr.open_dataset(f"{BASE}largerivers-100m{suffix}.nc"))
        drain_ds = slice_dataset(xr.open_dataset(f"{BASE}drains-100m{suffix}.nc"))
        tiledrain_ds = slice_dataset(xr.open_dataset(f"{BASE}tiledrainage-100m{suffix}.nc"))
        return [
            rsp.River.from_dataset(river_ds),
            rsp.River.from_dataset(large_river_ds),
            rsp.Drainage.from_dataset(drain_ds),
            rsp.Drainage.from_dataset(tiledrain_ds),
            overlandflow,
        ]

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(VARIANTS.keys())
    mae = np.full((len(labels), len(KD_VALUES)), np.nan)
    print(f"{len(labels)} variants x {len(KD_VALUES)} kD = "
          f"{len(labels) * len(KD_VALUES)} solves (noise-free), gamma={GAMMA:g}\n")

    for vi, (label, suffix) in enumerate(VARIANTS.items()):
        boundaries = build_boundaries(suffix)
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
                maxiter=100, relax=1.0,
            )
            inverse.formulate()
            inverse.nonlinear_solve()
            err = inverse.head.isel(layer=0) - modelhead
            mae[vi, ki] = float(abs(err).mean())
            # release PARDISO C-level memory (gc can't reach it)
            for obj in (inverse, gwf):
                ls = getattr(obj, "linearsolver", None)
                if ls is not None and hasattr(ls, "free_memory"):
                    ls.free_memory()
            del inverse, gwf, err, transmissivity, recharge
            gc.collect()
        print(f"  done: {label}")

    result = xr.Dataset(
        {"mae": (("variant", "kD"), mae)},
        coords={"variant": labels, "kD": KD_VALUES},
        attrs={"gamma": float(GAMMA), "recharge": float(RECHARGE),
               "seed": SEED, "n_piezometers": N_PIEZOMETERS, "noise": "none"},
    )
    result.to_netcdf(RUN_DIR / "conductance_kD_sweep.nc")
    print(f"\nSaved {RUN_DIR / 'conductance_kD_sweep.nc'}")

    print("\nOptimum kD per variant (near-optimal = within 1% of min error):")
    for vi, label in enumerate(labels):
        errs = mae[vi]
        i = int(np.nanargmin(errs))
        within = KD_VALUES[errs <= errs[i] * 1.01]
        edge = "  (AT RANGE EDGE)" if i in (0, len(KD_VALUES) - 1) else ""
        print(f"  {label:9s}: kD = {KD_VALUES[i]:7.0f}  (err {errs[i]:.4f}); "
              f"within-1%: {within.min():.0f}-{within.max():.0f}{edge}")
    return result


if __name__ == "__main__":
    run_experiment()
