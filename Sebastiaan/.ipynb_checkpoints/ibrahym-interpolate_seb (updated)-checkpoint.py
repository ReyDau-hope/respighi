# %%

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xugrid as xu
from shapely.geometry import Point

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
N_PIEZOMETERS = 500
TRANSMISSIVITY = 1000.00
RECHARGE = 0.0001
REG_WEIGHT = 1000
SEED = 12345

# %%


def slice_dataset(ds):
    return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))


head = xr.open_dataset("../../case/ibrahym/ibrahym-head-l1-100m.nc")["head"]
modelhead = slice_dataset(head.isel(time=-1)) #Previously named finalhead
drain_ds = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-drains-100m.nc"))
overlandflow_ds = slice_dataset(
    xr.open_dataset("../../case/ibrahym/ibrahym-overlandflow-100m.nc")
)
river_ds = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-rivers-100m.nc"))
large_river_ds = slice_dataset(
    xr.open_dataset("../../case/ibrahym/ibrahym-largerivers-100m.nc")
)
tiledrain_ds = slice_dataset(
    xr.open_dataset("../../case/ibrahym/ibrahym-tiledrainage-100m.nc")
)
subsoil = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-subsoil-100m.nc"))
hfb_gdf = gpd.read_file("../../case/ibrahym/hfb-12.gpkg") #Reads a GeoPackage file containing the horizontal flow barriers (HFBs) and creates a GeoDataFrame (gdf) from it.

# Select the winter data
river_ds = river_ds.isel(time=0)

# Create a transmissivity
transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), TRANSMISSIVITY)

# %%
# Initialize the relevant boundary condition classes, initialize the
# groundwater model, formulate, then solve.


transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), TRANSMISSIVITY)

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
)

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

# %%

#For loop reg_weight

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

# %%
#For loop kD values

kD_values = np.linspace(140, 8800, 20)
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

plt.figure()
plt.plot(kD_values, errors)
plt.xlabel("kD (m²/day)")
plt.ylabel("Mean absolute error (m)")
plt.title("Error vs Transmissivity")
plt.show()

# %%

# min_error_index = np.argmin(errors)
# min_kD = kD_values[min_error_index]
# min_error = errors[min_error_index]
# print(f"Optimal kD: {min_kD:.2f} m²/day, with mean error: {min_error:.4f} m")

sorted_indices_kD = np.argsort(errors)
top5_indices_kD = sorted_indices_kD[:5]
top5_kD = kD_values[top5_indices_kD]
top5_errors_kD = np.array(errors)[top5_indices_kD]

for i in range(5):
    print(f"Rank {i+1}: kD = {top5_kD[i]:.2f}, error = {top5_errors_kD[i]:.4f} m")
# %%
