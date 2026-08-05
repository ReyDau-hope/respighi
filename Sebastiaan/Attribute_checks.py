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
print("drains vars:", list(drain_ds.data_vars))
print("tiledrain vars:", list(tiledrain_ds.data_vars))
# conductance field -> resistance
C = drain_ds["conductance"]   # adjust name if different
print("conductance: min/max/mean:", float(C.min()), float(C.max()), float(C.mean()))
print("conductance units/attrs:", C.attrs)

# %%
# how many cells actually have drains (non-zero/non-nan conductance)?
drained = tiledrain_ds["conductance"].notnull() & (tiledrain_ds["conductance"] > 0)
n_drained = int(drained.sum())
n_total = int(tiledrain_ds["conductance"].size)
print(f"drained cells: {n_drained} / {n_total} = {100*n_drained/n_total:.1f}%")

# %%
C = drain_ds["conductance"]
print("mean:", float(C.mean()), "median:", float(C.median()))
Cnz = C.where(C > 0)
print("median of nonzero:", float(Cnz.median()))



# %%
