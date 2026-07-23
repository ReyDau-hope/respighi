#%%
import xarray as xr

ds = xr.open_dataset("../../case/ibrahym/ibrahym-largerivers-100m-cond2.nc")

print(ds)                      # whole structure: dims, coords, all variables
print(list(ds.data_vars))      # just the variable names
print(ds["conductance"])       # one variable's metadata

c = ds["conductance"]
c.values                       # the raw NumPy array (the actual numbers)
float(c.min()), float(c.max()), float(c.mean())   # quick stats (skip NaN by default)

c.isel(y=0, x=slice(0, 10)).values     # first 10 cells of the top row, by index
c.sel(x=195_000, y=360_000, method="nearest")   # value at real-world coordinates

c.plot()                       # <-- a colour map of the whole grid
# %%

import matplotlib.pyplot as plt

orig   = xr.open_dataset("../../case/ibrahym/ibrahym-largerivers-100m.nc")["conductance"]
scaled = xr.open_dataset("../../case/ibrahym/ibrahym-largerivers-100m-cond2.nc")["conductance"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
orig.plot(ax=ax1, vmin=0, vmax=3000);   ax1.set_title("original")
scaled.plot(ax=ax2, vmin=0, vmax=3000); ax2.set_title("x2")
# %%
orig   = xr.open_dataset("../../case/ibrahym/ibrahym-largerivers-100m.nc")["conductance"]
scaled = xr.open_dataset("../../case/ibrahym/ibrahym-largerivers-100m-cond0.5.nc")["conductance"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
orig.plot(ax=ax1, vmin=0, vmax=3000);   ax1.set_title("original")
scaled.plot(ax=ax2, vmin=0, vmax=3000); ax2.set_title("x0.5")
# %%
#Mostly redundant code below, but kept for reference. The only difference is the conductance value in the file name.

#Using original data
# head = xr.open_dataset("../../case/ibrahym/ibrahym-head-l1-100m.nc")["head"]
# modelhead = slice_dataset(head.isel(time=-1)) #Previously named finalhead
# drain_ds = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-drains-100m.nc"))
# overlandflow_ds = slice_dataset(
#     xr.open_dataset("../../case/ibrahym/ibrahym-overlandflow-100m.nc")
# )
# river_ds = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-rivers-100m.nc"))
# large_river_ds = slice_dataset(
#     xr.open_dataset("../../case/ibrahym/ibrahym-largerivers-100m.nc")
# )
# tiledrain_ds = slice_dataset(
#     xr.open_dataset("../../case/ibrahym/ibrahym-tiledrainage-100m.nc")
# )
# subsoil = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-subsoil-100m.nc"))
# hfb_gdf = gpd.read_file("../../case/ibrahym/hfb-12.gpkg") #Reads a GeoPackage file containing the horizontal flow barriers (HFBs) and creates a GeoDataFrame (gdf) from it.

#%%
#Using Conductance x0.5 data
# head = xr.open_dataset("../../case/ibrahym/ibrahym-head-l1-100m.nc")["head"]
# modelhead = slice_dataset(head.isel(time=-1)) #Previously named finalhead
# drain_ds = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-drains-100m-cond0.5.nc"))
# overlandflow_ds = slice_dataset(
#     xr.open_dataset("../../case/ibrahym/ibrahym-overlandflow-100m.nc")
# )
# river_ds = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-rivers-100m-cond0.5.nc"))
# large_river_ds = slice_dataset(
#     xr.open_dataset("../../case/ibrahym/ibrahym-largerivers-100m-cond0.5.nc")
# )
# tiledrain_ds = slice_dataset(
#     xr.open_dataset("../../case/ibrahym/ibrahym-tiledrainage-100m-cond0.5.nc")
# )
# subsoil = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-subsoil-100m.nc"))
# hfb_gdf = gpd.read_file("../../case/ibrahym/hfb-12.gpkg")

#%%
#Using Conductance x2 data
head = xr.open_dataset("../../case/ibrahym/ibrahym-head-l1-100m.nc")["head"]
modelhead = slice_dataset(head.isel(time=-1)) #Previously named finalhead
drain_ds = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-drains-100m-cond2.nc"))
overlandflow_ds = slice_dataset(
    xr.open_dataset("../../case/ibrahym/ibrahym-overlandflow-100m.nc")
)
river_ds = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-rivers-100m-cond2.nc"))
large_river_ds = slice_dataset(
    xr.open_dataset("../../case/ibrahym/ibrahym-largerivers-100m-cond2.nc")
)
tiledrain_ds = slice_dataset(
    xr.open_dataset("../../case/ibrahym/ibrahym-tiledrainage-100m-cond2.nc")
)
subsoil = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-subsoil-100m.nc"))
hfb_gdf = gpd.read_file("../../case/ibrahym/hfb-12.gpkg")

#%%
#Using Conductance x3 data
# head = xr.open_dataset("../../case/ibrahym/ibrahym-head-l1-100m.nc")["head"]
# modelhead = slice_dataset(head.isel(time=-1)) #Previously named finalhead
# drain_ds = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-drains-100m-cond3.nc"))
# overlandflow_ds = slice_dataset(
#     xr.open_dataset("../../case/ibrahym/ibrahym-overlandflow-100m.nc")
# )
# river_ds = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-rivers-100m-cond3.nc"))
# large_river_ds = slice_dataset(
#     xr.open_dataset("../../case/ibrahym/ibrahym-largerivers-100m-cond3.nc")
# )
# tiledrain_ds = slice_dataset(
#     xr.open_dataset("../../case/ibrahym/ibrahym-tiledrainage-100m-cond3.nc")
# )
# subsoil = slice_dataset(xr.open_dataset("../../case/ibrahym/ibrahym-subsoil-100m.nc"))
# hfb_gdf = gpd.read_file("../../case/ibrahym/hfb-12.gpkg")