"""
3x3 sigma-misspecification experiment for Respighi inverse head interpolation.

Crosses the *injected* noise level (sigma_ext -- added to the synthetic
piezometer heads here, in this script) against the *assumed* noise level
(sigma_int -- handed to Respighi via the CellSampling `sigma` argument).

    sigma_ext  ("external")  = real corruption of the data. Respighi never sees
                               this number, only its effect baked into the heads.
    sigma_int  ("internal")  = the trust knob Respighi actually reads. Scalar
                               here -> mathematically a rescaling of the
                               regularization weight, so REG IS HELD FIXED across
                               the grid (vary it too and you confound the two).

Accuracy metric = MAE(inverse.head, modelhead) -- the IBRAHYM head is the truth,
exactly as in your existing runs. There is NO clean recharge truth in this setup
(recharge is the estimated unknown), so recharge is SAVED every run for the
histogram/spatial analysis, not scored.

Each of the 9 cells is solved, scored against the CLEAN modelhead (never the
noisy data), and saved to netCDF. Diagonal cells (sigma_ext == sigma_int) are the
"honest" runs; off-diagonal cells are the deliberate mismatch.

Only `build_experiment_inputs()` touches Respighi. Verify the 3 lines flagged
"<-- VERIFY" against your for-loop's inverse call, then run.
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

# Injected noise std, metres  (external / true corruption, added in this script)
SIGMA_EXT = [0.10, 0.20, 0.50]          # 10, 20, 50 cm
# Assumed noise std, metres   (internal / told value, passed to Respighi)
SIGMA_INT = [0.10, 0.20, 0.50]          # 10, 20, 50 cm

N_REPEATS = 3        # noise realisations averaged per cell (Huite: 2-3 is fine)

SAVE = True
SAVE_ROOT = Path("../SavedData")
# Timestamped subfolder so reruns don't clobber. Point at SAVE_ROOT directly if
# you'd rather keep one flat folder.
RUN_DIR = SAVE_ROOT / f"sigma3x3_{datetime.now():%Y%m%d_%H%M}"


# ---------------------------------------------------------------------------
# ADAPTER  --  the only part that touches Respighi
# ---------------------------------------------------------------------------
def build_experiment_inputs():
    """
    Your setup, run ONCE, plus a run_inverse closure the engine calls per cell.

    Returns
    -------
    clean_piezo_head : (N,) ndarray   clean heads at piezometers (noise added to this)
    clean_head_field : xr.DataArray   modelhead -- the truth for head MAE
    run_inverse      : callable(noisy_piezo_head, sigma_int) -> inverse object
    """
    import geopandas as gpd
    import xugrid as xu
    import respighi as rsp

    # ---- window + params (from your script) ----
    XMIN, XMAX = 185_000.0, 205_000.0
    YMIN, YMAX = 350_000.0, 370_000.0
    N_PIEZOMETERS = 200
    TRANSMISSIVITY = 1000.00
    RECHARGE = 0.001                     # placeholder; the inverse overwrites it
    SCENARIO = ""                        # "", "-cond0.5", "-cond2", "-cond3"
    BASE = "../case/ibrahym/ibrahym-"

    def slice_dataset(ds):
        return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))

    # ---- load truth + boundary data ----
    head = xr.open_dataset(f"{BASE}head-l1-100m.nc")["head"]
    modelhead = slice_dataset(head.isel(time=-1))            # <-- TRUTH (head)
    drain_ds        = slice_dataset(xr.open_dataset(f"{BASE}drains-100m{SCENARIO}.nc"))
    river_ds        = slice_dataset(xr.open_dataset(f"{BASE}rivers-100m{SCENARIO}.nc"))
    large_river_ds  = slice_dataset(xr.open_dataset(f"{BASE}largerivers-100m{SCENARIO}.nc"))
    tiledrain_ds    = slice_dataset(xr.open_dataset(f"{BASE}tiledrainage-100m{SCENARIO}.nc"))
    overlandflow_ds = slice_dataset(xr.open_dataset(f"{BASE}overlandflow-100m.nc"))
    subsoil         = slice_dataset(xr.open_dataset(f"{BASE}subsoil-100m.nc"))
    hfb_gdf         = gpd.read_file(f"{BASE.replace('ibrahym-', '')}hfb-12.gpkg")

    river_ds = river_ds.isel(time=0)
    transmissivity = xr.full_like(
        subsoil["kh"].isel(layer=0, drop=True), TRANSMISSIVITY
    )

    # ---- boundary conditions + forward model ----
    river = rsp.River.from_dataset(river_ds)
    large_river = rsp.River.from_dataset(large_river_ds)
    drain = rsp.Drainage.from_dataset(drain_ds)
    tiledrain = rsp.Drainage.from_dataset(tiledrain_ds)
    overlandflow = rsp.Drainage.from_dataset(
        overlandflow_ds, constant_conductance=500.0
    )
    recharge = rsp.Recharge(rate=xr.full_like(transmissivity, RECHARGE).to_numpy())
    hfb = rsp.HorizontalFlowBarrier.from_geodataframe(
        layer=0, barriers=hfb_gdf, template=transmissivity, max_snap_distance=10.0,
    )

    gwf = rsp.GroundwaterModel(
        area=100.0 * 100.0,
        initial=modelhead,
        recharge=recharge,
        head_boundaries=[river, large_river, drain, tiledrain, overlandflow],
        transmissivity=transmissivity,
        horizontal_flow_barriers=[hfb],
        xclose=1e-6,
        maxiter=50,
    )
    gwf.formulate()
    gwf.nonlinear_solve()

    # ---- piezometers (same locations as your seeded set) + clean heads ----
    rng_pz = np.random.default_rng(seed=SEED)
    px = XMIN + (XMAX - XMIN) * rng_pz.random(N_PIEZOMETERS)
    py = YMIN + (YMAX - YMIN) * rng_pz.random(N_PIEZOMETERS)

    grid = xu.Ugrid2d.from_structured(modelhead)
    clean_piezo_head = modelhead.sel(
        x=xr.DataArray(px), y=xr.DataArray(py), method="nearest"
    ).to_numpy()

    # ---- fixed regularization, IDENTICAL for every cell ----
    # <-- VERIFY: post-rename, how does your for-loop set regularization?
    #     Scalar path (confirmed to still exist):
    REG_WEIGHT = 1000.0
    #     ...or the object path from tikhonov.py, e.g.
    #        REGULARIZATION = rsp.UnscaledMinimumCurvature(1000.0)
    #        REGULARIZATION = rsp.MinimumCurvature(roughness_scale=...)
    #     Whichever it is, define it here so it never varies across the 9 cells.

    def run_inverse(noisy_piezo_head, sigma_int):
        target = rsp.CellSampling(
            px, py, noisy_piezo_head, grid,
            sigma=sigma_int,               # <-- the whole experiment lives here
        )
        inverse = rsp.InverseProblem(
            gwf, target,
            regularization_weight=REG_WEIGHT,   # <-- VERIFY param name post-rename
            maxiter=100, relax=0.0,
        )
        inverse.solve()                         # <-- VERIFY: .solve()/.nonlinear_solve()
        return inverse

    return clean_piezo_head, modelhead, run_inverse


# ---------------------------------------------------------------------------
# ENGINE  --  no edits needed below
# ---------------------------------------------------------------------------
def _head_2d(inv_head: xr.DataArray) -> xr.DataArray:
    """Drop the layer dim so head compares against the 2D modelhead."""
    return inv_head.isel(layer=0) if "layer" in inv_head.dims else inv_head


def head_mae(inv_head: xr.DataArray, truth: xr.DataArray) -> float:
    """Spatial-mean absolute error over the full field, NaN cells skipped.

    Compared on values: both live on the same sliced grid, so cells align by
    position. Truth is the CLEAN modelhead -- never the noisy piezometer data.
    """
    a = np.asarray(_head_2d(inv_head).values, dtype=float)
    b = np.asarray(truth.values, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"head shape {a.shape} != truth shape {b.shape}")
    return float(np.nanmean(np.abs(a - b)))


def cell_filename(sigma_ext: float, sigma_int: float, repeat: int) -> str:
    """e.g. run_ext10cm_int20cm_rep0.nc -- cm as zero-padded ints, sorts cleanly."""
    ext_cm = int(round(sigma_ext * 100))
    int_cm = int(round(sigma_int * 100))
    return f"run_ext{ext_cm:02d}cm_int{int_cm:02d}cm_rep{repeat}.nc"


def save_run(inverse, sigma_ext, sigma_int, repeat, mae_h, out_dir):
    """Save recharge + head fields (paired) to one self-describing netCDF.
    Recharge is here for the histogram/spatial analysis; head MAE is stamped in
    attrs so the matrix is recoverable and any cell replottable without solving."""
    ds = xr.Dataset(
        {"recharge": inverse.recharge, "head": inverse.head},
        attrs={
            "sigma_ext_m": float(sigma_ext),
            "sigma_int_m": float(sigma_int),
            "repeat": int(repeat),
            "honest": int(sigma_ext == sigma_int),
            "mae_head": float(mae_h),
            "seed": SEED,
            "note": "ext=injected noise std (m); int=assumed sigma told to Respighi",
        },
    )
    path = out_dir / cell_filename(sigma_ext, sigma_int, repeat)
    ds.to_netcdf(path)
    return path


def run_experiment():
    import gc

    clean_piezo_head, modelhead, run_inverse = build_experiment_inputs()
    clean_piezo_head = np.asarray(clean_piezo_head, dtype=float)

    if SAVE:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving runs to: {RUN_DIR.resolve()}\n")

    rng = np.random.default_rng(SEED)
    n_ext, n_int = len(SIGMA_EXT), len(SIGMA_INT)
    mae_cube = np.full((n_ext, n_int, N_REPEATS), np.nan)

    for i, sigma_ext in enumerate(SIGMA_EXT):
        for rep in range(N_REPEATS):
            # Corrupt ONCE per (ext, repeat), then reuse across all sigma_int:
            # within a repeat, only trust changes down the row, not the data.
            noisy = clean_piezo_head + rng.normal(
                0.0, sigma_ext, size=clean_piezo_head.shape
            )

            for j, sigma_int in enumerate(SIGMA_INT):
                inverse = run_inverse(noisy, sigma_int)
                mae_h = head_mae(inverse.head, modelhead)
                mae_cube[i, j, rep] = mae_h

                if SAVE:
                    save_run(inverse, sigma_ext, sigma_int, rep, mae_h, RUN_DIR)

                tag = "honest" if sigma_ext == sigma_int else "mismatch"
                print(f"  ext={sigma_ext:.2f}  int={sigma_int:.2f}  rep={rep}  "
                      f"MAE_head={mae_h:.4e}   [{tag}]")

                del inverse           # release solver memory before the next cell
                gc.collect()

    # Average the METRIC over repeats (not the fields) -> the 3x3 matrix.
    mae_matrix = np.nanmean(mae_cube, axis=2)

    result = xr.Dataset(
        {"mae_head": (("sigma_ext", "sigma_int"), mae_matrix)},
        coords={"sigma_ext": SIGMA_EXT, "sigma_int": SIGMA_INT},
        attrs={"n_repeats": N_REPEATS, "seed": SEED,
               "metric": "MAE(inverse.head, modelhead)"},
    )
    if SAVE:
        result.to_netcdf(RUN_DIR / "mae_matrix_3x3.nc")

    print("\nMean head MAE  (rows = sigma_ext, cols = sigma_int):")
    print(np.array_str(mae_matrix, precision=4))
    return result


def plot_heatmap(result, out_dir=None):
    """3x3 heatmap with the honest diagonal outlined in red."""
    import matplotlib.pyplot as plt

    m = result["mae_head"].values
    ext = result["sigma_ext"].values
    intv = result["sigma_int"].values

    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    im = ax.imshow(m, origin="upper", cmap="viridis")
    ax.set_xticks(range(len(intv)), [f"{v:.2f}" for v in intv])
    ax.set_yticks(range(len(ext)), [f"{v:.2f}" for v in ext])
    ax.set_xlabel(r"$\sigma_{\mathrm{int}}$ (told) [m]")
    ax.set_ylabel(r"$\sigma_{\mathrm{ext}}$ (injected) [m]")
    ax.set_title("head MAE   (diagonal = honest)")

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=9,
                        color="white" if im.norm(val) < 0.6 else "black")
    for k in range(min(m.shape)):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="red", lw=2))

    fig.colorbar(im, ax=ax, label="MAE(head, modelhead) [m]")
    fig.tight_layout()
    if out_dir is not None:
        fig.savefig(Path(out_dir) / "head_mae_heatmap.png", dpi=150)
    return fig


if __name__ == "__main__":
    result = run_experiment()
    try:
        import matplotlib.pyplot as plt
        plot_heatmap(result, RUN_DIR if SAVE else None)
        plt.show()
    except Exception as exc:
        print(f"(plotting skipped: {exc})")
