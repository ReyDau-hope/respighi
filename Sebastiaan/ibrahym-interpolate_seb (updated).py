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
TRANSMISSIVITY = 4000.00
RECHARGE = 0.001
REG_WEIGHT = 10
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

# %%
BASE = "../../case/ibrahym/ibrahym-"

for name in ["rivers", "largerivers", "drains", "tiledrainage"]:
    orig = xr.open_dataset(f"{BASE}{name}-100m.nc")["conductance"]
    print(f"\n{name}")
    print(f"  original : min {float(orig.min()):.4f}  max {float(orig.max()):.4f}  mean {float(orig.mean()):.4f}")
    for factor in ["0.5", "2", "3"]:
        scaled = xr.open_dataset(f"{BASE}{name}-100m-cond{factor}.nc")["conductance"]
        ratio = float(scaled.mean()) / float(orig.mean())
        print(f"  x{factor:4s}    : min {float(scaled.min()):.4f}  max {float(scaled.max()):.4f}  "
              f"mean {float(scaled.mean()):.4f}  (ratio {ratio:.4f})")

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

#########################################RESPIGHI END###########################################
# %%

#######################======FOR LOOP REG WEIGHT ============####################

# reg_values = np.logspace(-1, 6, 50)
# errors_reg = []

# for reg in reg_values:
#     inverse = rsp.InverseProblem(
#         groundwatermodel=gwf,
#         target=target,
#         regularization_weight=reg,
#         maxiter=100,
#         relax=0.0,
#     )
#     inverse.formulate()
#     inverse.nonlinear_solve()
#     inversehead = inverse.head.isel(layer=0)
#     error = inversehead - modelhead
#     errors_reg.append(abs(error).mean().values)

# plt.figure()
# plt.plot(reg_values, errors_reg)
# plt.xscale("log")
# plt.xlabel("Regularization weight")
# plt.ylabel("Mean absolute error (m)")
# plt.title("Error vs Regularization Weight")
# plt.show()

# %%

# Find the reg value with the lowest error (only one value, not the top 5)
# min_error_index_reg = np.argmin(errors_reg)
# min_reg = reg_values[min_error_index_reg]
# min_error_reg = errors_reg[min_error_index_reg]
# print(f"Optimal reg: {min_reg:.2f}, with mean error: {min_error_reg:.4f} m")


#Find the top 5 reg values with the lowest error
# sorted_indices_reg = np.argsort(errors_reg)
# top5_indices_reg = sorted_indices_reg[:5]
# top5_reg = reg_values[top5_indices_reg]
# top5_errors_reg = np.array(errors_reg)[top5_indices_reg]

# for i in range(5):
#     print(f"Rank {i+1}: reg = {top5_reg[i]:.2f}, error = {top5_errors_reg[i]:.4f} m")

# %%

#Determining layer thickness and transmissivity

thickness = subsoil.top - subsoil.bottom
kD_per_layer = subsoil.kh * thickness
kD_total = kD_per_layer.sum(dim="layer")
print(kD_total)

# %%
print("dims:", dict(subsoil.sizes))
print("layers:", subsoil.layer.values)
print("variables:", list(subsoil.data_vars))
print("x:", float(subsoil.x.min()), "-", float(subsoil.x.max()), "| n =", subsoil.x.size)
print("y:", float(subsoil.y.min()), "-", float(subsoil.y.max()), "| n =", subsoil.y.size)
print()
print(subsoil)

# %%
####============ FOR LOOP FINDING OPTIMUM KD VALUE =============######

# kD_values = np.linspace(140, 8800, 20)
# errors = []

# for kD in kD_values:
#     transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), kD)
#     recharge = rsp.Recharge(
#         rate=xr.full_like(transmissivity, RECHARGE).to_numpy(),
#     )
#     gwf = rsp.GroundwaterModel(
#         area=100.0 * 100.0,
#         initial=modelhead,
#         recharge=recharge,
#         head_boundaries=[river, large_river, drain, tiledrain, overlandflow],
#         transmissivity=transmissivity,
#         horizontal_flow_barriers=[hfb],
#         xclose=1e-6,
#         maxiter=50,
#     )
#     gwf.formulate()
#     gwf.nonlinear_solve()
#     inverse = rsp.InverseProblem(
#         groundwatermodel=gwf,
#         target=target,
#         regularization_weight=REG_WEIGHT,
#         maxiter=100,
#         relax=0.0,
#     )
#     inverse.formulate()
#     inverse.nonlinear_solve()
#     inversehead = inverse.head.isel(layer=0)
#     error = inversehead - modelhead
#     errors.append(abs(error).mean().values)

# # Linear plot
# plt.figure(figsize=(8, 6))
# plt.plot(kD_values, errors, marker=".")
# plt.xlabel("kD (m²/day)")
# plt.ylabel("Mean absolute error (m)")
# plt.title("Error vs Transmissivity")
# plt.show()

# # Log plot
# plt.figure(figsize=(8, 6))
# plt.plot(kD_values, errors, marker=".")
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Transmissivity (m²/d)")
# plt.ylabel("Mean absolute error (m)")
# plt.title("Error vs Transmissivity (log-log)")
# plt.show()


# %%
# Overlay error-vs-kD curves across conductance variants
# PREREQ: run these upstream once (from ORIGINAL data) so they exist in the session:
#   modelhead, subsoil, hfb, target   (target built from original head + piezometers)

BASE = "../../case/ibrahym/ibrahym-"

# variant label -> suffix on the conductance-bearing files ("" = original)
variants = {
    "Original":  "",
    "cond x0.5": "-cond0.5",
    "cond x2":   "-cond2",
    "cond x3":   "-cond3",
}

kD_values = np.logspace(np.log10(1000), np.log10(10000), 40)
results = {}   # label -> list of errors

for label, suffix in variants.items():
    # conductance-bearing datasets for THIS variant
    river_ds        = slice_dataset(xr.open_dataset(f"{BASE}rivers-100m{suffix}.nc")).isel(time=0)
    large_river_ds  = slice_dataset(xr.open_dataset(f"{BASE}largerivers-100m{suffix}.nc"))
    drain_ds        = slice_dataset(xr.open_dataset(f"{BASE}drains-100m{suffix}.nc"))
    tiledrain_ds    = slice_dataset(xr.open_dataset(f"{BASE}tiledrainage-100m{suffix}.nc"))
    overlandflow_ds = slice_dataset(xr.open_dataset(f"{BASE}overlandflow-100m.nc"))  # always original

    # rebuild boundary objects from these datasets
    river        = rsp.River.from_dataset(river_ds)
    large_river  = rsp.River.from_dataset(large_river_ds)
    drain        = rsp.Drainage.from_dataset(drain_ds)
    tiledrain    = rsp.Drainage.from_dataset(tiledrain_ds)
    overlandflow = rsp.Drainage.from_dataset(overlandflow_ds, constant_conductance=500.0)

    errors = []
    for kD in kD_values:
        transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), kD)
        recharge = rsp.Recharge(rate=xr.full_like(transmissivity, RECHARGE).to_numpy())
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
            groundwatermodel=gwf, target=target,
            regularization_weight=REG_WEIGHT, maxiter=100, relax=0.0,
        )
        inverse.formulate()
        inverse.nonlinear_solve()
        inversehead = inverse.head.isel(layer=0)
        error = inversehead - modelhead
        errors.append(abs(error).mean().values)
        # release Pardiso's C-level MKL memory (gc.collect can't reach it)
        for obj in (inverse, gwf):
            ls = getattr(obj, "linearsolver", None)
            if ls is not None and hasattr(ls, "free_memory"):
                ls.free_memory()
        del inverse, gwf, inversehead, error, transmissivity, recharge
        gc.collect()

    results[label] = errors
    print(f"done: {label}")

# %%
# Overlay plot
plt.figure(figsize=(9, 6))
for label, errors in results.items():
    plt.plot(kD_values, errors, marker=".", label=label)
plt.xscale("log")
plt.xlabel("kD (m²/day)")
plt.ylabel("Mean absolute error (m)")
plt.title("Error vs Transmissivity — conductance variants")
plt.legend()
plt.show()


# %%
# Original-only kD sweep, log axes (matches the old presentation graph)
plt.figure(figsize=(8, 6))
plt.plot(kD_values, results["Original"], marker=".")
plt.xscale("log")
plt.yscale("log")     # image 2 had log y too; drop this line for linear y like image 3
plt.xlabel("Transmissivity (m²/d)")
plt.ylabel("Mean absolute error (m)")
plt.title("Error vs Transmissivity — original data")
plt.show()
# %%
# Optimum kD per conductance scenario
print("Optimum kD (minimum error) per scenario:")
for label, errs in results.items():
    errs = np.array(errs)
    i = int(np.argmin(errs))
    print(f"  {label:9s}: kD = {kD_values[i]:8.1f} m²/day   (error = {errs[i]:.4f} m)")
# %%
# Optimum kD + how flat the minimum is (kD within 1% of min error)
print("Optimum kD and near-optimal range per scenario:")
for label, errs in results.items():
    errs = np.array(errs)
    i = int(np.argmin(errs))
    emin = errs[i]
    within = kD_values[errs <= emin * 1.01]   # kD values within 1% of the min
    print(f"  {label:9s}: opt kD = {kD_values[i]:8.1f} m²/day  (err {emin:.4f} m); "
          f"range {within.min():.0f}–{within.max():.0f}")

#%%
# %%
# Run parameters (for record-keeping)
print("=== Run parameters ===")
#print(f"kD sweep      : {kD_values.min():.0f} – {kD_values.max():.0f} m²/day, {len(kD_values)} points (log-spaced)")
print(f"N_PIEZOMETERS : {N_PIEZOMETERS}")
print(f"TRANSMISSIVITY: {TRANSMISSIVITY}")
print(f"RECHARGE      : {RECHARGE}")
print(f"REG_WEIGHT    : {REG_WEIGHT}")
print(f"SEED          : {SEED}")
#print(f"variants      : {list(variants.keys())}")
print("======================")
# %%
# Numbers behind shift-vs-deform: optimal kD + min error per variant
# for label, errors in results.items():
#     i = int(np.argmin(errors))
#     print(f"{label:9s}: optimal kD = {kD_values[i]:7.1f} m²/day, min error = {errors[i]:.4f} m")
# %%

# min_error_index = np.argmin(errors)
# min_kD = kD_values[min_error_index]
# min_error = errors[min_error_index]
# print(f"Optimal kD: {min_kD:.2f} m²/day, with mean error: {min_error:.4f} m")

# sorted_indices_kD = np.argsort(errors)
# top5_indices_kD = sorted_indices_kD[:5]
# top5_kD = kD_values[top5_indices_kD]
# top5_errors_kD = np.array(errors)[top5_indices_kD]

# for i in range(5):
#     print(f"Rank {i+1}: kD = {top5_kD[i]:.2f}, error = {top5_errors_kD[i]:.4f} m")
# %%
print("inverse:", [a for a in dir(inverse) if 'solver' in a.lower() or 'free' in a.lower()])
print("gwf:", [a for a in dir(gwf) if 'solver' in a.lower() or 'free' in a.lower()])
# %%
