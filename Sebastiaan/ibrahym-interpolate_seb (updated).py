# %%

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xugrid as xu
from shapely.geometry import Point
import gc

import respighi as rsp

# %%
# Set window for area of interest

XMIN = 185_000.0
XMAX = 205_000.0
YMIN = 350_000.0
YMAX = 370_000.0
# XMIN = 170_000.0
# XMAX = XMIN + 15_000.0
# YMIN = 360_000.0
# YMAX = YMIN + 15_000.0

#Define parameters
N_PIEZOMETERS = 200
TRANSMISSIVITY = 2000.00
RECHARGE = 0.001
REG_WEIGHT = rsp.UnscaledMinimumCurvature(4000.0)
SEED = 12345


# %%

def slice_dataset(ds):
    return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))

SCENARIO = ""   # "" for original, "-cond0.5", "-cond2", "-cond3"

BASE = "../case/ibrahym/ibrahym-"

head = xr.open_dataset(f"{BASE}head-l1-100m.nc")["head"]
modelhead = slice_dataset(head.isel(time=-1))
drain_ds        = slice_dataset(xr.open_dataset(f"{BASE}drains-100m{SCENARIO}.nc"))
river_ds        = slice_dataset(xr.open_dataset(f"{BASE}rivers-100m{SCENARIO}.nc"))
large_river_ds  = slice_dataset(xr.open_dataset(f"{BASE}largerivers-100m{SCENARIO}.nc"))
tiledrain_ds    = slice_dataset(xr.open_dataset(f"{BASE}tiledrainage-100m{SCENARIO}.nc"))
overlandflow_ds = slice_dataset(xr.open_dataset(f"{BASE}overlandflow-100m.nc"))  # always original
subsoil         = slice_dataset(xr.open_dataset(f"{BASE}subsoil-100m.nc"))       # always original
hfb_gdf         = gpd.read_file(f"{BASE.replace('ibrahym-', '')}hfb-12.gpkg")

# Select the winter data
river_ds = river_ds.isel(time=0)

# Create a transmissivity
transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), TRANSMISSIVITY)

# %%
#Check conductance values of the boundary conditions

# base = "../../case/ibrahym/"

# for name in ["rivers", "largerivers", "drains", "tiledrainage"]:
#     orig = xr.open_dataset(f"{base}ibrahym-{name}-100m.nc")["conductance"]
#     for factor in [0.5, 2, 3]:
#         scaled = xr.open_dataset(f"{base}ibrahym-{name}-100m-cond{factor}.nc")["conductance"]
#         # Compare only where the original is non-NaN; check scaled == orig * factor
#         ok = np.allclose(scaled.values, orig.values * factor, equal_nan=True)
#         print(f"{name:12s} factor {factor}: exact match = {ok}")

# %%
# Initialize the relevant boundary condition classes, initialize the
# groundwater model, formulate, then solve.

river = rsp.River.from_dataset(river_ds)
large_river = rsp.River.from_dataset(large_river_ds)
drain = rsp.Drainage.from_dataset(drain_ds)
tiledrain = rsp.Drainage.from_dataset(tiledrain_ds)
overlandflow = rsp.Drainage.from_dataset(overlandflow_ds, constant_conductance=500.0)
recharge = rsp.Recharge(
    rate=xr.full_like(transmissivity, RECHARGE).to_numpy(),
)

hfb = rsp.HorizontalFlowBarrier.from_geodataframe(
    layer=0,
    barriers=hfb_gdf,
    template=transmissivity,
    max_snap_distance=10.0,
) #can be skipped

# # %%
# BASE = "../../case/ibrahym/ibrahym-"

# for name in ["rivers", "largerivers", "drains", "tiledrainage"]:
#     orig = xr.open_dataset(f"{BASE}{name}-100m.nc")["conductance"]
#     print(f"\n{name}")
#     print(f"  original : min {float(orig.min()):.4f}  max {float(orig.max()):.4f}  mean {float(orig.mean()):.4f}")
#     for factor in ["0.5", "2", "3"]:
#         scaled = xr.open_dataset(f"{BASE}{name}-100m-cond{factor}.nc")["conductance"]
#         ratio = float(scaled.mean()) / float(orig.mean())
#         print(f"  x{factor:4s}    : min {float(scaled.min()):.4f}  max {float(scaled.max()):.4f}  "
#               f"mean {float(scaled.mean()):.4f}  (ratio {ratio:.4f})")

# %%

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
gwf.head.isel(layer=0).plot.contour(levels=30)

# %%
# Make synthetic piezometers


def make_piezometers(n_piezometers, xmin, xmax, ymin, ymax):
    rng = np.random.default_rng(seed=SEED)
    x = xmin + (xmax - xmin) * rng.random(n_piezometers)
    y = ymin + (ymax - ymin) * rng.random(n_piezometers)
    return x, y

#Loading fixed piezometer locations (Cannot be used in conjunction with make_piezometers function, 
# as it will overwrite the generated piezometer locations with the ones from the file)
# piezometers_gdf = gpd.read_file(f"piezometers_seed{SEED}.gpkg")
# x = piezometers_gdf.geometry.x.to_numpy()
# y = piezometers_gdf.geometry.y.to_numpy()


x, y = make_piezometers(
    n_piezometers=200,
    xmin=XMIN,
    xmax=XMAX,
    ymin=YMIN,
    ymax=YMAX,
)

#Store fixed piezometer locations in a GeoPackage file for future reference
geometry = [Point(xi, yi) for xi, yi in zip(x, y)]
piezometers_gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:28992")
piezometers_gdf.to_file(f"piezometers_seed{SEED}.gpkg", driver="GPKG")

# %%
# Check where they are located
fig, ax = plt.subplots()
modelhead.plot(ax=ax)
ax.scatter(x, y, color="k", alpha=0.5)

# %%
# Make a respighi Target

grid = xu.Ugrid2d.from_structured(modelhead)
headvalues = modelhead.sel(
    x=xr.DataArray(x), y=xr.DataArray(y), method="nearest"
).to_numpy()
target = rsp.CellSampling(x, y, headvalues, grid)

# %%
# Inverse Problem
# ---------------
#
# With the groundwater model and the target, we can pose an inverse problem to solve.

inverse = rsp.InverseProblem(
    groundwatermodel=gwf,
    target=target,
    regularization=REG_WEIGHT,
    maxiter=100,
    relax=1.0,
)

# %%

inverse.formulate()

# %%
# Solve.

inverse.nonlinear_solve()
# %%

inversehead = inverse.head.isel(layer=0)
inversehead.plot.imshow()
# %%

error = inversehead - modelhead
print(abs(error).mean())

# %%

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 7))
modelhead.plot.contour(ax=ax0, levels=30)
inversehead.plot.contour(ax=ax1, levels=30)
ax0.set_aspect(1.0)
ax1.set_aspect(1.0)

# %%

error.plot.imshow(levels=np.arange(-1.0, 1.0, 0.1))
print(abs(error).mean())
print([a for a in dir(inverse) if 'recharge' in a.lower() or 'rate' in a.lower()])
print(float(inverse.recharge.mean()))

#########################################RESPIGHI END###########################################

# %%
# ================== SOLVE + SAVE RUN (data, plots, convergence) ==================
import contextlib, io
from pathlib import Path
from datetime import datetime

# Capture the per-iteration maxdh stream (the solver only PRINTS it, so grab stdout).
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    converged, n_iter = inverse.nonlinear_solve()
maxdh = [float(v) for v in _buf.getvalue().split() if v.strip()]
print(f"converged={converged}, iters={n_iter}"
      + (f", final maxdh={maxdh[-1]:.2e}" if maxdh else ""))

inversehead = inverse.head.isel(layer=0)
error = inversehead - modelhead

reg_value = REG_WEIGHT.weight
scenario_label = SCENARIO.lstrip("-") if SCENARIO else "original"
mae = float(abs(error).mean())

RUN_DIR = Path("../SavedData") / (
    f"run_{datetime.now():%Y%m%d_%H%M%S}"
    f"_kD{int(TRANSMISSIVITY)}_reg{int(reg_value)}_{scenario_label}"
)
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ---- convergence trajectory -> text file + log-y plot ----
np.savetxt(RUN_DIR / "convergence.txt", np.asarray(maxdh))
if maxdh:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(range(1, len(maxdh) + 1), maxdh, marker=".")
    ax.axhline(1e-4, color="red", ls="--", label="tolerance 1e-4")
    ax.set_xlabel("iteration"); ax.set_ylabel("maxdh (m)")
    ax.set_title(f"Convergence  (converged={converged}, iters={n_iter})")
    ax.legend()
    fig.savefig(RUN_DIR / "convergence.png", dpi=200, bbox_inches="tight")

# ---- DATA ----
def _strip_time(da):
    return da.drop_vars("time") if "time" in da.coords else da

ds = xr.Dataset(
    {
        "inverse_head": _strip_time(inverse.head),
        "recharge":     _strip_time(inverse.recharge),
        "modelhead":    _strip_time(modelhead),
        "error":        _strip_time(error),
    },
    attrs={
        "kD": float(TRANSMISSIVITY),
        "reg_weight": float(reg_value),
        "recharge_input": float(RECHARGE),
        "scenario": scenario_label,
        "seed": int(SEED),
        "n_piezometers": int(N_PIEZOMETERS),
        "mae_head": mae,
        "converged": int(converged),
        "n_iterations": int(n_iter),
        "final_maxdh": float(maxdh[-1]) if maxdh else float("nan"),
        "created": datetime.now().isoformat(timespec="seconds"),
    },
)
ds.to_netcdf(RUN_DIR / "run_data.nc")
np.savez(RUN_DIR / "piezometers.npz", x=x, y=y, headvalues=headvalues)

# ---- PLOT: error map ----
fig, ax = plt.subplots(figsize=(7, 6))
error.plot.imshow(ax=ax, levels=np.arange(-1.0, 1.0, 0.1))
ax.set_aspect("equal"); ax.set_title(f"Head error (MAE = {mae:.4f} m)")
fig.savefig(RUN_DIR / "error_map.png", dpi=200, bbox_inches="tight")

# ---- PLOT: truth vs interpolated ----
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 7))
modelhead.plot.contour(ax=ax0, levels=30)
inversehead.plot.contour(ax=ax1, levels=30)
ax0.set_title("IBRAHYM (truth)"); ax1.set_title("Respighi (interpolated)")
ax0.set_aspect(1.0); ax1.set_aspect(1.0)
fig.savefig(RUN_DIR / "head_contours.png", dpi=200, bbox_inches="tight")

# ---- PLOT: recharge ----
rech = np.asarray(inverse.recharge).ravel()
rech = rech[np.isfinite(rech)]
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
ax0.hist(rech, bins=60, edgecolor="black", alpha=0.8)
ax0.axvline(np.mean(rech), color="red", ls="--", label=f"mean = {np.mean(rech):.2e}")
ax0.axvline(np.median(rech), color="orange", ls="--", label=f"median = {np.median(rech):.2e}")
ax0.axvline(RECHARGE, color="green", ls=":", label=f"input = {RECHARGE:.2e}")
ax0.set_xlabel("Fitted recharge (m/day)"); ax0.set_ylabel("cells")
ax0.set_title("Recharge distribution"); ax0.legend()
vmax = float(np.nanmax(np.abs(rech)))
inverse.recharge.plot.imshow(ax=ax1, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    cbar_kwargs={"label": "Fitted recharge (m/day)"})
ax1.set_aspect("equal"); ax1.set_title("Recharge spatial distribution")
fig.suptitle(f"kD={TRANSMISSIVITY:.0f}, reg={reg_value:.0f}, {scenario_label}, seed={SEED}")
fig.savefig(RUN_DIR / "recharge.png", dpi=200, bbox_inches="tight")

# ---- summary ----
(RUN_DIR / "run_summary.txt").write_text(
    f"kD              : {TRANSMISSIVITY:.1f} m2/d\n"
    f"reg_weight      : {reg_value:.1f}\n"
    f"recharge_input  : {RECHARGE:.2e} m/d\n"
    f"scenario        : {scenario_label}\n"
    f"seed            : {SEED}\n"
    f"n_piezometers   : {N_PIEZOMETERS}\n"
    f"MAE (head)      : {mae:.6f} m\n"
    f"converged       : {converged}\n"
    f"n_iterations    : {n_iter}\n"
    f"final_maxdh     : {maxdh[-1] if maxdh else float('nan'):.6e} m\n"
    f"recharge_mean   : {float(inverse.recharge.mean()):.6e} m/d\n"
)
print(f"Saved run to: {RUN_DIR.resolve()}")
# %%
