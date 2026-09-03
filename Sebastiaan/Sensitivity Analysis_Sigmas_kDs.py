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
TRANSMISSIVITY = 1000.00
RECHARGE = 0.001
REG_WEIGHT = 10
SEED = 12345


# %%

def slice_dataset(ds):
    return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))

SCENARIO = ""   # "" for original, "-cond0.5", "-cond2", "-cond3"

BASE = r"C:\Users\sebas\Documents\1Thesis\case\ibrahym\ibrahym-"

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
####==== EXPERIMENT B: optimum kD re-found at each noise level ====####
# For each sigma, sweep kD, find the kD that minimises error.
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent.parent / "Findings"
OUTDIR.mkdir(parents=True, exist_ok=True)

scenario_label = SCENARIO.lstrip("-") if SCENARIO else "original"

sigma_cm  = np.array([10, 20, 50])
sigma_m   = sigma_cm / 100.0
kD_values = np.linspace(1000, 10000, 20) #normal values (1000, 4000, 12) kD value range and no. of steps
N_REPEATS = 5     # noise realisations per (sigma, kD), averaged

# clean sampled heads at piezometer locations (placement fixed by SEED upstream)
clean_heads = modelhead.sel(x=xr.DataArray(x), y=xr.DataArray(y), method="nearest").to_numpy()

optimum_per_sigma = []   # (sigma_cm, opt_kD, opt_error)
all_curves = {}          # sigma_cm -> mean error curve over kD

for s_cm, s in zip(sigma_cm, sigma_m):
    errors_kD = []
    for kD in kD_values:
        # build model at this kD (independent of noise)
        transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), kD)
        recharge = rsp.Recharge(rate=xr.full_like(transmissivity, RECHARGE).to_numpy())
        gwf = rsp.GroundwaterModel(
            area=100.0*100.0, initial=modelhead, recharge=recharge,
            head_boundaries=[river, large_river, drain, tiledrain, overlandflow],
            transmissivity=transmissivity, horizontal_flow_barriers=[hfb],
            xclose=1e-6, maxiter=50,
        )
        gwf.formulate()
        gwf.nonlinear_solve()

        # average error over several noise realisations
        reps = []
        for r in range(N_REPEATS):
            rng_noise = np.random.default_rng(seed=1000 + r)
            noisy = clean_heads + rng_noise.normal(0.0, s, size=clean_heads.shape)
            target = rsp.CellSampling(x, y, noisy, grid)
            inv = rsp.InverseProblem(groundwatermodel=gwf, target=target,
                                     regularization=REG_WEIGHT, maxiter=100, relax=1.0)
            inv.formulate()
            inv.nonlinear_solve()
            err = inv.head.isel(layer=0) - modelhead
            reps.append(float(abs(err).mean()))
            inv.linearsolver.free_memory()
            del inv; gc.collect()

        errors_kD.append(np.mean(reps))
        del gwf; gc.collect()

    errors_kD = np.array(errors_kD)
    i = int(np.argmin(errors_kD))
    within = kD_values[errors_kD <= errors_kD[i] * 1.02]
    print(f"sigma {s_cm:2d} cm: opt kD = {kD_values[i]:7.1f}  error {errors_kD[i]:.4f}  "
          f"| within-2% plateau: {within.min():.0f}–{within.max():.0f}  ({within.size} pts)")
    optimum_per_sigma.append((s_cm, kD_values[i], errors_kD[i]))
    all_curves[s_cm] = errors_kD
    print(f"sigma {s_cm:2d} cm: optimum kD = {kD_values[i]:7.1f} m²/day  (error {errors_kD[i]:.4f} m)")

# %%
# Overlay the kD-error curves per sigma
plt.figure(figsize=(9,6))
for s_cm, curve in all_curves.items():
    plt.plot(kD_values, curve, marker=".", label=f"σ = {s_cm} cm")
plt.xscale("log")
plt.xlabel("kD (m²/day)")
plt.ylabel("Mean absolute error (m)")
plt.title("Error vs kD at increasing measurement noise")
plt.legend()
plt.savefig(OUTDIR / f"NoiseTest_{int(TRANSMISSIVITY)}_reg{REG_WEIGHT}_{scenario_label}_SEED{SEED}.png",
            dpi=200, bbox_inches="tight")
plt.show()

# %%
# Optimum kD as a function of noise
opt_sigma = [o[0] for o in optimum_per_sigma]
opt_kD    = [o[1] for o in optimum_per_sigma]
plt.figure(figsize=(8,6))
plt.plot(opt_sigma, opt_kD, marker="o")
plt.xlabel("Measurement noise σ (cm)")
plt.ylabel("Optimum kD (m²/day)")
plt.title("Does the optimum kD shift with measurement noise?")
plt.savefig(OUTDIR / f"kD(function of noise)_{int(TRANSMISSIVITY)}_reg{REG_WEIGHT}_{scenario_label}_SEED{SEED}.png",
            dpi=200, bbox_inches="tight")
plt.show()

# %%
# Save
with open("experimentB_optimum_vs_noise.txt", "a") as f:
    f.write(f"# SEED={SEED}, REG_WEIGHT={REG_WEIGHT}, N_REPEATS={N_REPEATS}, kD grid {kD_values.min():.0f}-{kD_values.max():.0f}\n")
    for s_cm, okd, oerr in optimum_per_sigma:
        f.write(f"sigma_cm={s_cm}, opt_kD={okd:.1f}, error={oerr:.4f}\n")