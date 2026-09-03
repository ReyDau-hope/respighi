"""
Experiment B: optimum kD at each measurement-noise level.

For each injected noise sigma (10/20/50 cm), sweeps kD, averages MAE over
N_REPEATS noise realisations, and finds the optimum kD -- to test whether the
optimum kD shifts with noise. Current API (regularization object, relax=1.0),
convergence captured, saves everything to SavedData.

Companion plotter (curves + optimum-vs-noise): plot_kD_vs_noise.py. RE-POINT BASE.
"""

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
GAMMA = 4000.0                    # regularization weight (was REG_WEIGHT)
RECHARGE = 0.001
N_PIEZOMETERS = 200
N_REPEATS = 5                     # noise realisations per (sigma, kD), averaged

SIGMA_CM = np.array([10, 20, 50])                 # injected noise levels, cm
KD_VALUES = np.logspace(np.log10(500), np.log10(10000), 20)

SCENARIO = ""                     # "", "-cond0.5", "-cond2", "-cond3"
XMIN, XMAX = 185_000.0, 205_000.0
YMIN, YMAX = 350_000.0, 370_000.0
BASE = r"C:\Users\sebas\Documents\1Thesis\case\ibrahym\ibrahym-"   # <-- RE-POINT

SAVE_ROOT = Path("../SavedData")
scenario_label = SCENARIO.lstrip("-") if SCENARIO else "original"
RUN_DIR = SAVE_ROOT / f"kD_vs_noise_{scenario_label}_{datetime.now():%Y%m%d_%H%M}"


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
    clean_heads = modelhead.sel(
        x=xr.DataArray(px), y=xr.DataArray(py), method="nearest"
    ).to_numpy()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    n_s, n_k = len(SIGMA_CM), len(KD_VALUES)
    mae = np.full((n_s, n_k), np.nan)            # mean over repeats
    frac_conv = np.full((n_s, n_k), np.nan)      # fraction of repeats that converged
    print(f"{n_s} sigma x {n_k} kD x {N_REPEATS} reps = {n_s*n_k*N_REPEATS} solves, "
          f"gamma={GAMMA:g}, scenario={scenario_label}\n")

    for si, s_cm in enumerate(SIGMA_CM):
        s = s_cm / 100.0
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

            reps, conv_flags = [], []
            for r in range(N_REPEATS):
                rng_noise = np.random.default_rng(seed=1000 + r)
                noisy = clean_heads + rng_noise.normal(0.0, s, size=clean_heads.shape)
                target = rsp.CellSampling(px, py, noisy, grid)
                inv = rsp.InverseProblem(
                    gwf, target, regularization=rsp.UnscaledMinimumCurvature(GAMMA),
                    maxiter=100, relax=1.0,
                )
                inv.formulate()
                conv, _ = inv.nonlinear_solve()
                err = inv.head.isel(layer=0) - modelhead
                reps.append(float(abs(err).mean()))
                conv_flags.append(int(conv))
                ls = getattr(inv, "linearsolver", None)
                if ls is not None and hasattr(ls, "free_memory"):
                    ls.free_memory()
                del inv; gc.collect()

            mae[si, ki] = float(np.mean(reps))
            frac_conv[si, ki] = float(np.mean(conv_flags))
            ls = getattr(gwf, "linearsolver", None)
            if ls is not None and hasattr(ls, "free_memory"):
                ls.free_memory()
            del gwf, transmissivity, recharge; gc.collect()

        # optimum kD for this sigma (converged-heavy cells only)
        ok = frac_conv[si] >= 0.5
        errs = np.where(ok, mae[si], np.inf)
        i = int(np.nanargmin(errs))
        print(f"sigma {s_cm:2d} cm: opt kD = {KD_VALUES[i]:7.0f}  "
              f"MAE {mae[si, i]:.4f}  (conv frac {frac_conv[si, i]:.1f})")

    # ---- save ----
    result = xr.Dataset(
        {"mae": (("sigma_cm", "kD"), mae),
         "frac_converged": (("sigma_cm", "kD"), frac_conv)},
        coords={"sigma_cm": SIGMA_CM, "kD": KD_VALUES},
        attrs={"gamma": float(GAMMA), "recharge": float(RECHARGE), "seed": SEED,
               "n_repeats": N_REPEATS, "scenario": scenario_label},
    )
    result.to_netcdf(RUN_DIR / "kD_vs_noise.nc")
    print(f"\nSaved {RUN_DIR / 'kD_vs_noise.nc'}")

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = plt.cm.tab10(range(n_s))
    for si, s_cm in enumerate(SIGMA_CM):
        line, = ax.plot(KD_VALUES, mae[si], marker=".", color=colors[si],
                        label=f"$\\sigma$ = {s_cm} cm")
        ok = frac_conv[si] >= 0.5
        j = int(np.nanargmin(np.where(ok, mae[si], np.inf)))
        ax.plot(KD_VALUES[j], mae[si][j], "*", ms=14, color=line.get_color())
    ax.set_xscale("log")
    ax.set_xlabel(r"kD (m$^2$/day)")
    ax.set_ylabel("mean absolute error (m)")
    ax.set_title(f"Error vs kD at increasing noise  ($\\gamma$ = {GAMMA:g}, {scenario_label})\n"
                 "$\\star$ = optimum per noise level")
    ax.legend()
    fig.savefig(RUN_DIR / "kD_vs_noise_curves.png", dpi=200, bbox_inches="tight")

    opt_kD = [KD_VALUES[int(np.nanargmin(np.where(frac_conv[si] >= 0.5, mae[si], np.inf)))]
              for si in range(n_s)]
    fig2, ax2 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax2.plot(SIGMA_CM, opt_kD, marker="o")
    ax2.set_xlabel(r"measurement noise $\sigma$ (cm)")
    ax2.set_ylabel(r"optimum kD (m$^2$/day)")
    ax2.set_title("Does the optimum kD shift with measurement noise?")
    fig2.savefig(RUN_DIR / "kD_vs_noise_optimum.png", dpi=200, bbox_inches="tight")

    print(f"Saved plots to {RUN_DIR.resolve()}")
    return result


if __name__ == "__main__":
    run_experiment()
    plt.show()
