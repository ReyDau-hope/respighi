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
TRANSMISSIVITY = 2200.00
RECHARGE = 0.0001
REG_WEIGHT = 10000
SEED = 12345


# %%

def slice_dataset(ds):
    return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))

SCENARIO = ""   # "" for original, "-cond0.5", "-cond2", "-cond3"

BASE = "../../case/ibrahym/ibrahym-"

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
    regularization_weight=REG_WEIGHT,
    maxiter=100,
    relax=0.0,
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

# %%
# Save run parameters + fitted recharge stats
scenario_label = SCENARIO if SCENARIO else "original"

with open("recharge_per_scenario.txt", "a") as f:
    f.write(
        f"scenario={scenario_label}, "
        f"SEED={SEED}, "
        f"REG_WEIGHT={REG_WEIGHT}, "
        f"TRANSMISSIVITY={TRANSMISSIVITY}, "
        f"recharge_mean={float(inverse.recharge.mean()):.6e}, "
        f"recharge_median={float(np.median(np.asarray(inverse.recharge))):.6e}\n"
    )

print("saved:", scenario_label)

# %%
# Fitted recharge: histogram + spatial map  (run after a SINGLE inverse solve)
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent.parent / "Findings"
OUTDIR.mkdir(parents=True, exist_ok=True)

scenario_label = SCENARIO.lstrip("-") if SCENARIO else "original"

rech_da = inverse.recharge
rech = np.asarray(rech_da).ravel()
rech = rech[np.isfinite(rech)]

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 6))

ax0.hist(rech, bins=60, edgecolor="black", alpha=0.8)
ax0.axvline(np.mean(rech),   color="red",    linestyle="--", label=f"mean = {np.mean(rech):.2e}")
ax0.axvline(np.median(rech), color="orange", linestyle="--", label=f"median = {np.median(rech):.2e}")
ax0.axvline(RECHARGE,        color="green",  linestyle=":",  label=f"input = {RECHARGE:.2e}")
ax0.set_xlabel("Fitted recharge (m/day)")
ax0.set_ylabel("Number of cells")
ax0.set_title("Distribution of fitted recharge")
ax0.legend()

vmax = np.nanmax(np.abs(rech))
rech_da.plot.imshow(
    ax=ax1, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    cbar_kwargs={"label": "Fitted recharge (m/day)"}, add_labels=True,
)
ax1.set_aspect("equal")
ax1.set_title("Spatial distribution of fitted recharge")

plt.suptitle(f"kD = {TRANSMISSIVITY:.0f}, reg_w = {REG_WEIGHT}, conductance = {scenario_label}", y=1.02)
plt.tight_layout()
plt.savefig(OUTDIR / f"recharge_kD{int(TRANSMISSIVITY)}_reg{REG_WEIGHT}_{scenario_label}_SEED{SEED}.png",
            dpi=200, bbox_inches="tight")
plt.show()
# %%
####============ FOR LOOP FINDING OPTIMUM KD VALUE =============######

kD_values = np.linspace(1000, 10000, 20)
errors = []

for kD in kD_values:
    transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), kD)
    recharge = rsp.Recharge(
        rate=xr.full_like(transmissivity, RECHARGE).to_numpy(),
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
    inverse = rsp.InverseProblem(
        groundwatermodel=gwf,
        target=target,
        regularization_weight=REG_WEIGHT,
        maxiter=100,
        relax=0.0,
    )
    inverse.formulate()
    inverse.nonlinear_solve()
    inversehead = inverse.head.isel(layer=0)
    error = inversehead - modelhead
    errors.append(abs(error).mean().values)

#%%
# Linear plot
plt.figure(figsize=(8, 6))
plt.plot(kD_values, errors, marker=".")
plt.xlabel("kD (m²/day)")
plt.ylabel("Mean absolute error (m)")
plt.title(f"Error vs Transmissivity - {scenario_label}")
plt.savefig(OUTDIR / f"LinearPlot_kD{int(TRANSMISSIVITY)}_reg{REG_WEIGHT}_{scenario_label}_SEED{SEED}.png",
            dpi=200, bbox_inches="tight")
plt.show()

# Log plot
plt.figure(figsize=(8, 6))
plt.plot(kD_values, errors, marker=".")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Transmissivity (m²/d)")
plt.ylabel("Mean absolute error (m)")
plt.title(f"Error vs Transmissivity (log-log) - {scenario_label}")
plt.savefig(OUTDIR / f"LogPLot_kD{int(TRANSMISSIVITY)}_reg{REG_WEIGHT}_{scenario_label}_SEED{SEED}.png",
            dpi=200, bbox_inches="tight")
plt.show()

# %%
##===============LARGELY REDUNDANT DUE TO ADDITION OF HISTOGRAM==========##
# Save ranked kD values (best-fit first) with run parameters
# scenario_label = SCENARIO if SCENARIO else "original"

# # pair each kD with its error, sort ascending by error
# ranked = sorted(zip(kD_values, errors), key=lambda pair: pair[1])

# with open("ranked_kD.txt", "a") as f:
#     f.write(
#         f"# scenario={scenario_label}, SEED={SEED}, REG_WEIGHT={REG_WEIGHT}, "
#         f"N_REPEATS={N_REPEATS if 'N_REPEATS' in dir() else 'NA'}, "
#         f"kD grid {kD_values.min():.0f}-{kD_values.max():.0f}, {len(kD_values)} pts\n"
#     )
#     for rank, (kd, err) in enumerate(ranked[:5], start=1):
#         f.write(f"rank {rank:2d}: kD={kd:8.1f}, error={err:.6f}\n")
#     f.write("\n")   # blank line between runs

# print("saved ranking:", scenario_label)

# %%

min_error_index = np.argmin(errors)
min_kD = kD_values[min_error_index]
min_error = errors[min_error_index]
print(f"[{scenario_label}] Optimal kD: {min_kD:.2f} m²/day, with mean error: {min_error:.4f} m")

sorted_indices_kD = np.argsort(errors)
top5_indices_kD = sorted_indices_kD[:5]
top5_kD = kD_values[top5_indices_kD]
top5_errors_kD = np.array(errors)[top5_indices_kD]

for i in range(5):
    print(f"[{scenario_label}] Rank {i+1}: kD = {top5_kD[i]:.2f}, error = {top5_errors_kD[i]:.4f} m")



# %%
