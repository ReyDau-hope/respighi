"""
Regularization-weight sweep over the 3x3 sigma-misspecification experiment.

For each regularization weight w, runs the full 3x3 (sigma_ext x sigma_int) grid
and records head MAE against the clean modelhead. Tests Huite's prediction: at
the correctly-tuned w, the honest diagonal (sigma_int == sigma_ext) becomes the
best cell in every row, and the best sigma_int stops drifting with injected error.

    under-regularized w  -> trusting data LESS always helps -> row-min at MAX sigma_int (right col)
    tuned w              -> honesty wins                    -> row-min ON the diagonal
    over-regularized w   -> trusting data MORE helps        -> row-min at MIN sigma_int (left col)

On disk:
    RUN_DIR/
        reg000500/  run_ext10cm_int20cm_rep0.nc ...   per-reg cells; the existing
        reg001000/  ...                               recharge_analysis.py / compare_cells.py
        ...                                           work when pointed at ONE reg folder
        regsweep_summary.nc                           MAE cube over (reg, ext, int)

Only build_experiment_inputs() touches Respighi. RE-POINT `BASE` to your data path
(the fix you already applied locally).
"""
#%%
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEED = 12345

SIGMA_EXT = [0.10, 0.20, 0.50]          # injected noise std, m (external/true)
SIGMA_INT = [0.10, 0.20, 0.50]          # assumed noise std, m (internal/told)

# Regularization weights to sweep. Brackets your current 1000 on both sides;
# Huite expects the optimum "a bit higher". Geometric spacing since it's a
# scale hyperparameter. Trim/extend freely -- see the solve-count note on run.
REG_WEIGHTS = [5500.0, 6000.0]

N_REPEATS = 3        # noise realisations averaged per cell (2-3 is fine)

SAVE = True
SAVE_ROOT = Path("../SavedData")
RUN_DIR = SAVE_ROOT / f"sigma_regsweep_{datetime.now():%Y%m%d_%H%M}"

#%%
# ---------------------------------------------------------------------------
# ADAPTER  --  the only part that touches Respighi
# ---------------------------------------------------------------------------
def build_experiment_inputs():
    """
    Your setup, run ONCE, plus a run_inverse closure the engine calls per cell.

    Returns
    -------
    clean_piezo_head : (N,) ndarray   clean heads at piezometers (noise added to this)
    clean_head_field : xr.DataArray   modelhead -- truth for head MAE
    run_inverse      : callable(noisy_piezo_head, sigma_int, reg_weight) -> inverse
    """
    import geopandas as gpd
    import xugrid as xu
    import respighi as rsp

    XMIN, XMAX = 185_000.0, 205_000.0
    YMIN, YMAX = 350_000.0, 370_000.0
    N_PIEZOMETERS = 200
    TRANSMISSIVITY = 2000.00
    RECHARGE = 0.001                     # m/d placeholder; the inverse overwrites it
    SCENARIO = ""
    BASE = "../case/ibrahym/ibrahym-"     # <-- RE-POINT to your data path

    def slice_dataset(ds):
        return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))

    head = xr.open_dataset(f"{BASE}head-l1-100m.nc")["head"]
    modelhead = slice_dataset(head.isel(time=-1))            # truth (head)
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

    river = rsp.River.from_dataset(river_ds)
    large_river = rsp.River.from_dataset(large_river_ds)
    drain = rsp.Drainage.from_dataset(drain_ds)
    tiledrain = rsp.Drainage.from_dataset(tiledrain_ds)
    overlandflow = rsp.Drainage.from_dataset(overlandflow_ds, constant_conductance=500.0)
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

    rng_pz = np.random.default_rng(seed=SEED)
    px = XMIN + (XMAX - XMIN) * rng_pz.random(N_PIEZOMETERS)
    py = YMIN + (YMAX - YMIN) * rng_pz.random(N_PIEZOMETERS)

    grid = xu.Ugrid2d.from_structured(modelhead)
    clean_piezo_head = modelhead.sel(
        x=xr.DataArray(px), y=xr.DataArray(py), method="nearest"
    ).to_numpy()

    def run_inverse(noisy_piezo_head, sigma_int, reg_weight):
        # Regularization is UnscaledMinimumCurvature(weight) == weight * graph_Laplacian,
        # the backwards-compat operator matching your old regularization_weight. weight
        # is the swept hyperparameter here; sigma_int is the trust knob (both vary).
        regularization = rsp.UnscaledMinimumCurvature(reg_weight)
        target = rsp.CellSampling(px, py, noisy_piezo_head, grid, sigma=sigma_int)
        inverse = rsp.InverseProblem(
            gwf, target, regularization=regularization, maxiter=100, relax=0.0,
        )
        inverse.formulate()
        inverse.nonlinear_solve()
        return inverse

    return clean_piezo_head, modelhead, run_inverse

#%%
# ---------------------------------------------------------------------------
# ENGINE  --  no edits needed below
# ---------------------------------------------------------------------------
def _head_2d(inv_head: xr.DataArray) -> xr.DataArray:
    return inv_head.isel(layer=0) if "layer" in inv_head.dims else inv_head


def head_mae(inv_head: xr.DataArray, truth: xr.DataArray) -> float:
    a = np.asarray(_head_2d(inv_head).values, dtype=float)
    b = np.asarray(truth.values, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"head shape {a.shape} != truth shape {b.shape}")
    return float(np.nanmean(np.abs(a - b)))


def reg_tag(w: float) -> str:
    """Folder name encoding the weight, zero-padded so it sorts: reg001000."""
    return f"reg{int(round(w)):06d}"


def cell_filename(sigma_ext: float, sigma_int: float, repeat: int) -> str:
    ext_cm, int_cm = int(round(sigma_ext * 100)), int(round(sigma_int * 100))
    return f"run_ext{ext_cm:02d}cm_int{int_cm:02d}cm_rep{repeat}.nc"


def save_run(inverse, reg_weight, sigma_ext, sigma_int, repeat, mae_h, reg_dir):
    ds = xr.Dataset(
        {"recharge": inverse.recharge, "head": inverse.head},
        attrs={
            "reg_weight": float(reg_weight),
            "sigma_ext_m": float(sigma_ext),
            "sigma_int_m": float(sigma_int),
            "repeat": int(repeat),
            "honest": int(sigma_ext == sigma_int),
            "mae_head": float(mae_h),
            "seed": SEED,
            "note": "ext=injected noise std (m); int=assumed sigma; reg=UnscaledMinimumCurvature weight",
        },
    )
    path = reg_dir / cell_filename(sigma_ext, sigma_int, repeat)
    ds.to_netcdf(path)
    return path


def diagnose(matrix, ext, intv):
    """Per row, does the honest (diagonal) cell win? Returns (rows, n_diag_wins).

    Diagonal col for row i = the sigma_int closest to sigma_ext[i]. gap >= 0 is
    how much the best (possibly-lying) cell beats the honest one; gap == 0 means
    honesty wins that row.
    """
    intv = np.asarray(intv, dtype=float)
    rows = []
    for i, e in enumerate(ext):
        dj = int(np.argmin(np.abs(intv - e)))       # honest column for this row
        aj = int(np.nanargmin(matrix[i]))           # actual best column
        rows.append({
            "sigma_ext": float(e),
            "best_int": float(intv[aj]),
            "diag_wins": aj == dj,
            "gap": float(matrix[i, dj] - matrix[i, aj]),
        })
    return rows, sum(r["diag_wins"] for r in rows)


def run_experiment():
    import gc

    clean_piezo_head, modelhead, run_inverse = build_experiment_inputs()
    clean_piezo_head = np.asarray(clean_piezo_head, dtype=float)

    n_reg, n_ext, n_int = len(REG_WEIGHTS), len(SIGMA_EXT), len(SIGMA_INT)
    total = n_reg * n_ext * n_int * N_REPEATS
    print(f"Sweep: {n_reg} reg x {n_ext} ext x {n_int} int x {N_REPEATS} reps "
          f"= {total} solves\n")

    if SAVE:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving to: {RUN_DIR.resolve()}\n")

    rng = np.random.default_rng(SEED)
    cube = np.full((n_reg, n_ext, n_int, N_REPEATS), np.nan)

    for k, w in enumerate(REG_WEIGHTS):
        reg_dir = RUN_DIR / reg_tag(w)
        if SAVE:
            reg_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- reg_weight = {w:g} ---")

        for i, sigma_ext in enumerate(SIGMA_EXT):
            for rep in range(N_REPEATS):
                # Corrupt ONCE per (ext, rep); reuse across sigma_int AND reg so
                # cells differ only by trust/reg, not by noise realisation.
                noisy = clean_piezo_head + rng.normal(
                    0.0, sigma_ext, size=clean_piezo_head.shape
                )
                for j, sigma_int in enumerate(SIGMA_INT):
                    inverse = run_inverse(noisy, sigma_int, w)
                    mae_h = head_mae(inverse.head, modelhead)
                    cube[k, i, j, rep] = mae_h
                    if SAVE:
                        save_run(inverse, w, sigma_ext, sigma_int, rep, mae_h, reg_dir)
                    del inverse
                    gc.collect()

        # per-reg 3x3 (averaged over reps) + diagnostic
        matrix = np.nanmean(cube[k], axis=2)
        rows, n_diag = diagnose(matrix, SIGMA_EXT, SIGMA_INT)
        best_cols = [f"{r['best_int']:.2f}" for r in rows]
        mean_gap = float(np.mean([r["gap"] for r in rows]))
        print("  3x3 MAE:\n" + np.array_str(matrix, precision=4))
        print(f"  row-min at sigma_int = [{', '.join(best_cols)}]  "
              f"(diagonal wins {n_diag}/{n_ext} rows, mean honesty gap {mean_gap:.4e})\n")

    mae_mean = np.nanmean(cube, axis=3)                 # (reg, ext, int)
    result = xr.Dataset(
        {"mae_head": (("reg_weight", "sigma_ext", "sigma_int"), mae_mean)},
        coords={"reg_weight": REG_WEIGHTS, "sigma_ext": SIGMA_EXT, "sigma_int": SIGMA_INT},
        attrs={"n_repeats": N_REPEATS, "seed": SEED,
               "metric": "MAE(inverse.head, modelhead)"},
    )
    if SAVE:
        result.to_netcdf(RUN_DIR / "regsweep_summary.nc")

    _print_verdict(result)
    return result


def _diag_and_min_curves(result):
    """Mean-over-rows honest (diagonal) MAE and best-case (row-min) MAE per reg."""
    ext = result["sigma_ext"].values
    intv = result["sigma_int"].values
    reg = result["reg_weight"].values
    M = result["mae_head"].values                       # (reg, ext, int)
    diag = np.full(len(reg), np.nan)
    rmin = np.full(len(reg), np.nan)
    for k in range(len(reg)):
        d = []
        for i, e in enumerate(ext):
            dj = int(np.argmin(np.abs(intv - e)))
            d.append(M[k, i, dj])
        diag[k] = np.mean(d)
        rmin[k] = np.mean([np.nanmin(M[k, i]) for i in range(len(ext))])
    return reg, diag, rmin


def _print_verdict(result):
    reg, diag, rmin = _diag_and_min_curves(result)
    gap = diag - rmin
    k_tuned = int(np.argmin(gap))
    k_best = int(np.argmin(diag))
    print("Verdict:")
    print(f"  honesty gap (honest - best) per reg: "
          f"{dict(zip([f'{w:g}' for w in reg], np.round(gap, 4)))}")
    print(f"  smallest honesty gap at reg_weight = {reg[k_tuned]:g} "
          f"(gap {gap[k_tuned]:.4e}) -> the diagonal-optimal weight")
    print(f"  lowest honest MAE  at reg_weight = {reg[k_best]:g} "
          f"({diag[k_best]:.4e} m)")


def plot_convergence(result, out_dir=None):
    """diagonal (honest) vs row-min (best-case) mean MAE across reg. Where they
    meet, lying stops helping -> tuned reg_weight."""
    import matplotlib.pyplot as plt
    reg, diag, rmin = _diag_and_min_curves(result)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.plot(reg, diag, "-o", label="honest (diagonal) mean MAE")
    ax.plot(reg, rmin, "-s", label="best-case (row-min) mean MAE")
    ax.set_xscale("log")
    ax.set_xlabel("regularization weight")
    ax.set_ylabel("mean head MAE [m]")
    ax.set_title("Where the curves meet, honesty costs nothing\n(the tuned reg_weight)")
    ax.legend()
    if out_dir is not None:
        fig.savefig(Path(out_dir) / "regsweep_convergence.png", dpi=150)
    return fig

#%%
def plot_reg_grid(result, out_dir=None):
    """Small-multiple 3x3 heatmaps, one per reg. A dot marks each row's best
    cell; red boxes mark the honest diagonal. Watch the dots march onto the
    diagonal as reg increases."""
    import matplotlib.pyplot as plt
    reg = result["reg_weight"].values
    ext = result["sigma_ext"].values
    intv = result["sigma_int"].values
    M = result["mae_head"].values

    ncol = min(len(reg), 3)
    nrow = int(np.ceil(len(reg) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow),
                             constrained_layout=True, squeeze=False)
    vmin, vmax = np.nanmin(M), np.nanmax(M)
    for k, w in enumerate(reg):
        ax = axes[k // ncol][k % ncol]
        m = M[k]
        im = ax.imshow(m, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(intv)), [f"{v:.2f}" for v in intv])
        ax.set_yticks(range(len(ext)), [f"{v:.2f}" for v in ext])
        ax.set_title(f"reg = {w:g}")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.3g}", ha="center", va="center", fontsize=7,
                        color="white" if im.norm(m[i, j]) < 0.6 else "black")
            aj = int(np.nanargmin(m[i]))                        # best cell -> dot
            ax.plot(aj, i, "o", mfc="none", mec="red", ms=16, mew=2)
        for t in range(min(m.shape)):                           # honest diagonal -> box
            ax.add_patch(plt.Rectangle((t - 0.5, t - 0.5), 1, 1, fill=False,
                                       edgecolor="white", lw=1, ls="--"))
    for k in range(len(reg), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(r"row-min (red $\circ$) marches onto the honest diagonal "
                 r"(dashed) as reg increases", fontsize=12)
    if out_dir is not None:
        fig.savefig(Path(out_dir) / "regsweep_grid.png", dpi=150)
    return fig


if __name__ == "__main__":
    result = run_experiment()
    try:
        import matplotlib.pyplot as plt
        plot_convergence(result, RUN_DIR if SAVE else None)
        plot_reg_grid(result, RUN_DIR if SAVE else None)
        plt.show()
    except Exception as exc:
        print(f"(plotting skipped: {exc})")
# %%
