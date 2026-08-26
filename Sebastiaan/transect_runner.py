"""
Transect experiment runner (noise-free, conceptual).

Sweeps ONE parameter (kD or regularization) with the other fixed, runs Respighi
on the clean sampled heads, and saves each fitted head field plus the IBRAHYM
truth into a dedicated subfolder. Companion plotter: plot_transect.py.

Set SWEEP = "kD"  -> vary transmissivity, reg fixed at REG_FIXED
    SWEEP = "reg" -> vary regularization, kD fixed at KD_FIXED

Noise-free (clean heads) so the figure isolates the parameter's effect on the
fitted head, with nothing confounded by noise. Only build_experiment_inputs()
touches Respighi. RE-POINT `BASE` to your data path.
"""
#%%
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
#%%
# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEED = 12345
SWEEP = "kD"                     # "kD" or "reg"

KD_FIXED = 2000.0                # used when SWEEP == "reg"
REG_FIXED = 4000.0              # used when SWEEP == "kD"
SIGMA_INT = 0.1                  # fixed assumed noise (data is clean; sets data/reg balance)

KD_VALUES = [250.0, 500.0, 1000.0, 2000.0, 4000.0]      # a handful, well-separated
REG_VALUES = [100.0, 500.0, 1000.0, 4000.0, 8000.0]

SAVE_ROOT = Path("../SavedData")
RUN_DIR = SAVE_ROOT / f"transect_{SWEEP}_{datetime.now():%Y%m%d_%H%M}"

#%%
# ---------------------------------------------------------------------------
# ADAPTER  --  the only part that touches Respighi
# ---------------------------------------------------------------------------
def build_experiment_inputs():
    import geopandas as gpd
    import xugrid as xu
    import respighi as rsp

    XMIN, XMAX = 185_000.0, 205_000.0
    YMIN, YMAX = 350_000.0, 370_000.0
    N_PIEZOMETERS = 200
    RECHARGE = 0.001
    SCENARIO = ""
    BASE = "../case/ibrahym/ibrahym-"        # <-- RE-POINT to your data path

    def slice_dataset(ds):
        return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))

    head = xr.open_dataset(f"{BASE}head-l1-100m.nc")["head"]
    modelhead = slice_dataset(head.isel(time=-1))
    drain_ds        = slice_dataset(xr.open_dataset(f"{BASE}drains-100m{SCENARIO}.nc"))
    river_ds        = slice_dataset(xr.open_dataset(f"{BASE}rivers-100m{SCENARIO}.nc"))
    large_river_ds  = slice_dataset(xr.open_dataset(f"{BASE}largerivers-100m{SCENARIO}.nc"))
    tiledrain_ds    = slice_dataset(xr.open_dataset(f"{BASE}tiledrainage-100m{SCENARIO}.nc"))
    overlandflow_ds = slice_dataset(xr.open_dataset(f"{BASE}overlandflow-100m.nc"))
    subsoil         = slice_dataset(xr.open_dataset(f"{BASE}subsoil-100m.nc"))
    hfb_gdf         = gpd.read_file(f"{BASE.replace('ibrahym-', '')}hfb-12.gpkg")

    river_ds = river_ds.isel(time=0)
    kh_template = subsoil["kh"].isel(layer=0, drop=True)

    river = rsp.River.from_dataset(river_ds)
    large_river = rsp.River.from_dataset(large_river_ds)
    drain = rsp.Drainage.from_dataset(drain_ds)
    tiledrain = rsp.Drainage.from_dataset(tiledrain_ds)
    overlandflow = rsp.Drainage.from_dataset(overlandflow_ds, constant_conductance=500.0)
    hfb = rsp.HorizontalFlowBarrier.from_geodataframe(
        layer=0, barriers=hfb_gdf,
        template=xr.full_like(kh_template, 1.0), max_snap_distance=10.0,
    )
    boundaries = [river, large_river, drain, tiledrain, overlandflow]

    rng_pz = np.random.default_rng(seed=SEED)
    px = XMIN + (XMAX - XMIN) * rng_pz.random(N_PIEZOMETERS)
    py = YMIN + (YMAX - YMIN) * rng_pz.random(N_PIEZOMETERS)
    grid = xu.Ugrid2d.from_structured(modelhead)
    clean_piezo_head = modelhead.sel(
        x=xr.DataArray(px), y=xr.DataArray(py), method="nearest"
    ).to_numpy()

    def make_gwf(kD):
        transmissivity = xr.full_like(kh_template, float(kD))
        recharge = rsp.Recharge(rate=xr.full_like(transmissivity, RECHARGE).to_numpy())
        gwf = rsp.GroundwaterModel(
            area=100.0 * 100.0, initial=modelhead, recharge=recharge,
            head_boundaries=boundaries, transmissivity=transmissivity,
            horizontal_flow_barriers=[hfb], xclose=1e-6, maxiter=50,
        )
        gwf.formulate()
        gwf.nonlinear_solve()
        return gwf

    def run_inverse(gwf, heads, sigma_int, reg_weight):
        regularization = rsp.UnscaledMinimumCurvature(float(reg_weight))
        target = rsp.CellSampling(px, py, heads, grid, sigma=sigma_int)
        inverse = rsp.InverseProblem(
            gwf, target, regularization=regularization, maxiter=100, relax=0.0,
        )
        inverse.formulate()
        inverse.nonlinear_solve()
        return inverse

    return clean_piezo_head, modelhead, make_gwf, run_inverse

#%%
# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------
def _head_2d(da):
    return da.isel(layer=0) if "layer" in da.dims else da


def run_experiment():
    import gc

    clean, modelhead, make_gwf, run_inverse = build_experiment_inputs()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # save the truth once
    _head_2d(modelhead).to_dataset(name="head").to_netcdf(RUN_DIR / "truth.nc")
    print(f"Saving to: {RUN_DIR.resolve()}")

    values = KD_VALUES if SWEEP == "kD" else REG_VALUES
    print(f"Sweeping {SWEEP} over {values}\n")

    gwf_fixed = None
    if SWEEP == "reg":
        gwf_fixed = make_gwf(KD_FIXED)          # built once; only reg varies

    for v in values:
        if SWEEP == "kD":
            gwf = make_gwf(v)
            inverse = run_inverse(gwf, clean, SIGMA_INT, REG_FIXED)
        else:
            gwf = gwf_fixed
            inverse = run_inverse(gwf, clean, SIGMA_INT, v)

        fitted = _head_2d(inverse.head)
        ds = xr.Dataset(
            {"head": fitted},
            attrs={"sweep": SWEEP, "value": float(v),
                   "kD": float(v if SWEEP == "kD" else KD_FIXED),
                   "reg_weight": float(v if SWEEP == "reg" else REG_FIXED),
                   "sigma_int": SIGMA_INT, "seed": SEED},
        )
        ds.to_netcdf(RUN_DIR / f"head_{SWEEP}{int(round(v)):05d}.nc")
        print(f"  {SWEEP} = {v:g}  saved")
        if SWEEP == "kD":
            del gwf
        del inverse
        gc.collect()

    print(f"\nDone. Now run plot_transect.py pointed at {RUN_DIR.name}")


if __name__ == "__main__":
    run_experiment()
