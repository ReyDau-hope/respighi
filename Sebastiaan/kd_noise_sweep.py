"""
kD sweep across noise levels, at the tuned regularization weight.

Matches the reg-sweep's noise handling (Gaussian noise ADDED to the piezometer
heads, averaged over N repeats) so the two experiments differ only in WHAT is
swept -- not in noise-vs-no-noise. For each injected-noise level sigma_ext, kD is
swept and the optimal kD (lowest head MAE) recorded, to test:

  (a) does the clean-data kD optimum (~900 at reg=4000) survive when noise is added?
  (b) does the optimal kD shift with noise level?
  (c) is it self-consistent with the kD=2000 assumed during regularization tuning?

Assumed noise sigma_int is set equal to sigma_ext (the honest / correctly-specified
case), so the kD-vs-noise relation is not confounded by misspecification. Set
SIGMA_INT_FIXED to a number to override with a single fixed assumed noise instead.

Only build_experiment_inputs() touches Respighi. RE-POINT `BASE` to your data path.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEED = 12345

REG_WEIGHT = 4000.0                       # the tuned weight, held fixed for all cells
SIGMA_EXT = [0.10, 0.20, 0.50]            # injected noise std, m
SIGMA_INT_FIXED = None                    # None -> sigma_int = sigma_ext (honest)
KD_VALUES = np.geomspace(50, 6000, 20)    # log-spaced; brackets ~900 and 2000
N_REPEATS = 3

SAVE = True
SAVE_ROOT = Path("../SavedData")
RUN_DIR = SAVE_ROOT / f"kdsweep_noise_{datetime.now():%Y%m%d_%H%M}"


# ---------------------------------------------------------------------------
# ADAPTER  --  the only part that touches Respighi
# ---------------------------------------------------------------------------
def build_experiment_inputs():
    """
    Setup run ONCE (data, boundaries, piezometers), plus two closures:
      make_gwf(kD)                 -> a formulated+solved GroundwaterModel at that kD
      run_inverse(gwf, noisy, si)  -> solved InverseProblem at reg=REG_WEIGHT, sigma=si
    Boundaries and the HFB do not depend on kD, so they are built once; only the
    transmissivity field and the model are rebuilt per kD.
    """
    import geopandas as gpd
    import xugrid as xu
    import respighi as rsp

    XMIN, XMAX = 185_000.0, 205_000.0
    YMIN, YMAX = 350_000.0, 370_000.0
    N_PIEZOMETERS = 200
    RECHARGE = 0.001                      # m/d placeholder; the inverse overwrites it
    SCENARIO = ""
    BASE = "../case/ibrahym/ibrahym-"     # <-- RE-POINT to your data path

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
    kh_template = subsoil["kh"].isel(layer=0, drop=True)   # grid template for full_like

    # kD-independent pieces -- built once
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
            area=100.0 * 100.0,
            initial=modelhead,
            recharge=recharge,
            head_boundaries=boundaries,
            transmissivity=transmissivity,
            horizontal_flow_barriers=[hfb],
            xclose=1e-6,
            maxiter=50,
        )
        gwf.formulate()
        gwf.nonlinear_solve()
        return gwf

    def run_inverse(gwf, noisy_piezo_head, sigma_int):
        regularization = rsp.UnscaledMinimumCurvature(REG_WEIGHT)
        target = rsp.CellSampling(px, py, noisy_piezo_head, grid, sigma=sigma_int)
        inverse = rsp.InverseProblem(
            gwf, target, regularization=regularization, maxiter=100, relax=0.0,
        )
        inverse.formulate()
        inverse.nonlinear_solve()
        return inverse

    return clean_piezo_head, modelhead, make_gwf, run_inverse


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------
def _head_2d(inv_head):
    return inv_head.isel(layer=0) if "layer" in inv_head.dims else inv_head


def head_mae(inv_head, truth):
    a = np.asarray(_head_2d(inv_head).values, dtype=float)
    b = np.asarray(truth.values, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"head shape {a.shape} != truth shape {b.shape}")
    return float(np.nanmean(np.abs(a - b)))


def run_experiment():
    import gc

    clean, modelhead, make_gwf, run_inverse = build_experiment_inputs()
    clean = np.asarray(clean, dtype=float)

    n_s, n_k = len(SIGMA_EXT), len(KD_VALUES)
    total = n_k * n_s * N_REPEATS
    print(f"Sweep: {n_k} kD x {n_s} sigma_ext x {N_REPEATS} reps = {total} inverse "
          f"solves (+{n_k} forward solves) at reg={REG_WEIGHT:g}\n")

    if SAVE:
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-generate the noisy datasets ONCE per (sigma_ext, rep) so the SAME noise
    # realisation is reused across every kD -- the kD comparison is then clean.
    rng = np.random.default_rng(SEED)
    noise = {}
    for i, se in enumerate(SIGMA_EXT):
        for rep in range(N_REPEATS):
            noise[(i, rep)] = clean + rng.normal(0.0, se, size=clean.shape)

    cube = np.full((n_s, n_k, N_REPEATS), np.nan)      # (sigma_ext, kD, rep)

    for ki, kD in enumerate(KD_VALUES):
        gwf = make_gwf(kD)                              # built ONCE per kD
        for i, se in enumerate(SIGMA_EXT):
            si = se if SIGMA_INT_FIXED is None else SIGMA_INT_FIXED
            for rep in range(N_REPEATS):
                inverse = run_inverse(gwf, noise[(i, rep)], si)
                cube[i, ki, rep] = head_mae(inverse.head, modelhead)
                del inverse
                gc.collect()
        del gwf
        gc.collect()
        print(f"  kD = {kD:7.1f}  done ({ki + 1}/{n_k})")

    mae = np.nanmean(cube, axis=2)                      # (sigma_ext, kD)

    result = xr.Dataset(
        {"mae_head": (("sigma_ext", "kD"), mae)},
        coords={"sigma_ext": SIGMA_EXT, "kD": KD_VALUES},
        attrs={"reg_weight": REG_WEIGHT, "n_repeats": N_REPEATS, "seed": SEED,
               "sigma_int": "sigma_ext (honest)" if SIGMA_INT_FIXED is None
               else str(SIGMA_INT_FIXED),
               "metric": "MAE(inverse.head, modelhead)"},
    )
    if SAVE:
        result.to_netcdf(RUN_DIR / "kdsweep_noise_summary.nc")

    _report(result)
    return result


def _report(result):
    kd = result["kD"].values
    mae = result["mae_head"].values
    print(f"\nOptimal kD per noise level (reg={REG_WEIGHT:g}):")
    for i, se in enumerate(result["sigma_ext"].values):
        order = np.argsort(mae[i])
        best = kd[order[0]]
        edge = " (AT RANGE EDGE -- extend the sweep)" if order[0] in (0, len(kd) - 1) else ""
        top3 = ", ".join(f"{kd[o]:.0f} ({mae[i, o]:.4f})" for o in order[:3])
        print(f"  sigma_ext={se:.2f}: optimum kD ~ {best:.0f} m2/d{edge}")
        print(f"      top 3 [kD (MAE)]: {top3}")
    print("\nCompare against: reg-tuning assumed kD=2000; clean-data sweep gave ~900.")
    print("If the optima here cluster far from 2000, kD and reg are not jointly "
          "identifiable (the degeneracy). If they sit near 2000, the pair is "
          "self-consistent. Watch whether the optimum SHIFTS across noise levels.")


def plot_curves(result, out_dir=None):
    import matplotlib.pyplot as plt
    kd = result["kD"].values
    mae = result["mae_head"].values
    sig = result["sigma_ext"].values

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for i, se in enumerate(sig):
        line, = ax.plot(kd, mae[i], marker=".", label=f"$\\sigma_{{ext}}$ = {se:.2f}")
        ki = int(np.nanargmin(mae[i]))
        ax.plot(kd[ki], mae[i][ki], "*", ms=15, color=line.get_color())
    ax.axvline(2000, color="gray", ls=":", lw=1, label="kD=2000 (reg-tuning assumption)")
    ax.set_xscale("log")
    ax.set_xlabel("kD (m$^2$/day)")
    ax.set_ylabel("mean head MAE (m)")
    ax.set_title(f"Error vs kD across noise levels (reg = {REG_WEIGHT:g}, honest $\\sigma_{{int}}$)\n"
                 "$\\star$ = optimum per noise level")
    ax.legend()
    if out_dir is not None:
        fig.savefig(Path(out_dir) / "kdsweep_noise_curves.png", dpi=150)
    return fig


if __name__ == "__main__":
    result = run_experiment()
    try:
        import matplotlib.pyplot as plt
        plot_curves(result, RUN_DIR if SAVE else None)
        plt.show()
    except Exception as exc:
        print(f"(plotting skipped: {exc})")
