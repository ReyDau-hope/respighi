# %%

import xarray as xr
import respighi as rsp
import numpy as np
import xugrid as xu
import matplotlib.pyplot as plt

# %%
print("start of run")
head = xr.open_dataset("../case/ibrahym/ibrahym-head-l1-100m.nc")["head"]
# %%

#Define the region size, in this case 20x20km. 185_000 just means 185000, the underscore is just
#a seperator, easier to keep track of the values.
XMIN = 185_000.0
XMAX = 205_000.0
YMIN = 350_000.0
YMAX = 370_000.0

#Define parameters
N_PIEZOMETERS = 200
TRANSMISSIVITY = 1000.00
RECHARGE = 0.0001
REG_WEIGHT = 10

#define a function which takes any dataset "ds" as input.
#.sel() is coordinate-based selection, selecting by actual coordinate values (meters in this case)
#rather than index positions.
#slice() is an existing Python command saying "everything between these two values"
def slice_dataset(ds):
    return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))


# Select the last time and slice it spatially
finalhead = slice_dataset(head.isel(time=-1))
# %%

drains = slice_dataset(xr.open_dataset("../case/ibrahym/ibrahym-drains-100m.nc"))
overlandflow_ds = slice_dataset(xr.open_dataset("../case/ibrahym/ibrahym-overlandflow-100m.nc"))
rivers = slice_dataset(xr.open_dataset("../case/ibrahym/ibrahym-rivers-100m.nc"))
large_rivers = slice_dataset(xr.open_dataset("../case/ibrahym/ibrahym-largerivers-100m.nc"))
subsoil = slice_dataset(xr.open_dataset("../case/ibrahym/ibrahym-subsoil-100m.nc"))
tiledrains = slice_dataset(xr.open_dataset("../case/ibrahym/ibrahym-tiledrainage-100m.nc"))
# %%

# Select the winter data
rivers = rivers.isel(time=0)

# %%
# Initialize the relevant boundary condition classes, initialize the
# groundwater model, formulate, then solve.
river = rsp.River(
    conductance=rivers["conductance"].fillna(0.0).to_numpy(),
    stage=rivers["stage"].fillna(0.0).to_numpy(),
    elevation=rivers["bottom_elevation"].fillna(0.0).to_numpy(),
)
large_river = rsp.River(
    conductance=large_rivers["conductance"].fillna(0.0).to_numpy(),
    stage=large_rivers["stage"].fillna(0.0).to_numpy(),
    elevation=large_rivers["bottom_elevation"].fillna(0.0).to_numpy(),
)
drain = rsp.Drainage(
    conductance=drains["conductance"].fillna(0.0).to_numpy(),
    elevation=drains["elevation"].fillna(0.0).to_numpy(),
)
tiledrain = rsp.Drainage(
    conductance=tiledrains["conductance"].fillna(0.0).to_numpy(),
    elevation=tiledrains["elevation"].fillna(0.0).to_numpy(),
)
overlandflow = rsp.Drainage(
    conductance=xr.full_like(overlandflow_ds["elevation"], 500.0).to_numpy(),
    elevation=overlandflow_ds["elevation"].to_numpy(),
)
transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), TRANSMISSIVITY)
recharge = rsp.Recharge(
    rate=xr.full_like(transmissivity, RECHARGE).to_numpy(),
)
gwf = rsp.GroundwaterModel(
    area=100.0 * 100.0,  # 100 m resolution instead of 25 m
    initial=finalhead.to_numpy(),
    recharge=recharge,
    head_boundaries=[river, large_river, drain, tiledrain, overlandflow],
    transmissivity=transmissivity.to_numpy(),
    xclose=1e-6,
    maxiter=50,
)
gwf.formulate()
gwf.nonlinear_solve()


# %%

testhead = finalhead.copy(data=gwf.head.reshape(finalhead.shape))

# %%

testhead.plot.contour(levels=30)
#sanity check

# %%

#downsampling, in this case essentially downsampling from 100m to 1km resolution
dx = 1000.0
dy = -dx
x_coarse = np.arange(XMIN, XMAX, dx) + 0.5 * dx
y_coarse = np.arange(YMAX, YMIN, dy) + 0.5 * dy
coarse_template = xr.DataArray(
    data=np.zeros((y_coarse.size, x_coarse.size)),
    coords={"y": y_coarse, "x": x_coarse},
    dims=("y", "x")
)

# %%

regridder = xu.OverlapRegridder(source=finalhead, target=coarse_template)
coarsehead = regridder.regrid(finalhead)
grid = xu.Ugrid2d.from_structured(finalhead)
target = rsp.ModelTarget(coarsehead, grid)


# Make synthetic piezometers

def make_piezometers(n_piezometers, xmin, xmax, ymin, ymax):
    x = xmin + (xmax - xmin) * np.random.rand(n_piezometers)
    y = ymin + (ymax - ymin) * np.random.rand(n_piezometers)
    return x, y

#Commented out for now as this gets called in the for loop anyway, so this standalone call is
#now redundant


x, y = make_piezometers(
    n_piezometers=100,
    xmin=XMIN,
    xmax=XMAX,
    ymin=YMIN,
    ymax=YMAX,
)


# %%
# Check where they are located: (Commented out for now, irrelevant with for loop active)

fig, ax = plt.subplots()
finalhead.plot(ax=ax)
ax.scatter(x, y, color="k", alpha=0.5)

# %%
# Make a respighi Target

grid = xu.Ugrid2d.from_structured(finalhead)
headvalues = finalhead.sel(x=xr.DataArray(x), y=xr.DataArray(y), method="nearest").to_numpy()
noise = np.random.normal(loc=0, scale=0.1, size=headvalues.shape) #Generating random noise
headvalues_noisy = headvalues + noise #Adding random noise to measurements
target = rsp.CellSampling(x, y, headvalues_noisy, grid)

# %%
# Inverse Problem
# ---------------
#
# With the groundwater model and the target, we can pose an inverse problem to solve.

inverse = rsp.InverseProblem(
    groundwatermodel=gwf,
    target=target,
    regularization_weight= REG_WEIGHT, #Have not yet found a sweetspot for this variable
    maxiter=100,
    relax=0.0,
)

# %%

inverse.formulate()

# %%
# Solve.

inverse.nonlinear_solve()
# %%

inversehead = finalhead.copy(data=inverse.head.reshape(finalhead.shape))

# %%

inversehead.plot.imshow()
# %%

error = inversehead - finalhead

# %%

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 7))
finalhead.plot.contour(ax=ax0, levels=30)
inversehead.plot.contour(ax=ax1, levels=30)
ax0.set_aspect(1.0)
ax1.set_aspect(1.0)

# %%

error.plot.imshow(levels=np.arange(-1.0, 1.0, 0.1))
print(abs(error).mean())


# %%

print("end of run")

# %%

#Determining layer thickness and transmissivity

# thickness = subsoil.top - subsoil.bottom
# kD_per_layer = subsoil.kh * thickness
# kD_total = kD_per_layer.sum(dim="layer")

# %%

#For loop kD values

# kD_values = np.linspace(140, 8800, 50)
# errors = []

# for kD in kD_values:
#     transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), kD)
#     recharge = rsp.Recharge(
#         rate=xr.full_like(transmissivity, RECHARGE).to_numpy(),
#     )
#     gwf = rsp.GroundwaterModel(
#         area=100.0 * 100.0,
#         initial=finalhead.to_numpy(),
#         recharge=recharge,
#         head_boundaries=[river, large_river, drain, tiledrain, overlandflow],
#         transmissivity=transmissivity.to_numpy(),
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
#     inversehead = finalhead.copy(data=inverse.head.reshape(finalhead.shape))
#     error = inversehead - finalhead
#     errors.append(abs(error).mean().values)

# plt.figure()
# plt.plot(kD_values, errors)
# plt.xlabel("kD (m²/day)")
# plt.ylabel("Mean absolute error (m)")
# plt.title("Error vs Transmissivity")
# plt.show()

# # %%

# #Extract smallest error with corresponding kD value:
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

#For loop reg_weight

reg_values = np.logspace(-1, 6, 50)
errors_reg = []

for reg in reg_values:
    inverse = rsp.InverseProblem(
        groundwatermodel=gwf,
        target=target,
        regularization_weight=reg,
        maxiter=100,
        relax=0.0,
    )
    inverse.formulate()
    inverse.nonlinear_solve()
    inversehead = finalhead.copy(data=inverse.head.reshape(finalhead.shape))
    error = inversehead - finalhead
    errors_reg.append(abs(error).mean().values)

plt.figure()
plt.plot(reg_values, errors_reg)
plt.xscale("log")
plt.xlabel("Regularization weight")
plt.ylabel("Mean absolute error (m)")
plt.title("Error vs Regularization Weight")
plt.show()

# %%

# min_error_index_reg = np.argmin(errors_reg)
# min_reg = reg_values[min_error_index_reg]
# min_error_reg = errors_reg[min_error_index_reg]
# print(f"Optimal reg: {min_reg:.2f}, with mean error: {min_error_reg:.4f} m")

sorted_indices_reg = np.argsort(errors_reg)
top5_indices_reg = sorted_indices_reg[:5]
top5_reg = reg_values[top5_indices_reg]
top5_errors_reg = np.array(errors_reg)[top5_indices_reg]

for i in range(5):
    print(f"Rank {i+1}: reg = {top5_reg[i]:.2f}, error = {top5_errors_reg[i]:.4f} m")

# %%

#CheatSheet
#Example: thickness.mean(dim=["x", "y"]).to_series()

#slice.dataset > spatially slices the data to your area of interest (XMIN, XMAX, YMIN, YMAX)
#it crops your dataset to just the 20x20km region you are interested in. Everything outside 
#of that gets discarded.

#the .to_numpy() commands are to convert the xarray metadata to just raw values which is what
#respighi wants to gobble up and consume to make those nice interpolated graphs.


# %%

###### Potential Code to loop the runs and store the graphs ######
# n_runs = 10
# inverseheads = []
# errors = []

# for i in range(n_runs):
#     x, y = make_piezometers(
#         n_piezometers=N_PIEZOMETERS,
#         xmin=XMIN,
#         xmax=XMAX,
#         ymin=YMIN,
#         ymax=YMAX,
#     )
#     grid = xu.Ugrid2d.from_structured(finalhead)
#     headvalues = finalhead.sel(x=xr.DataArray(x), y=xr.DataArray(y), method="nearest").to_numpy()
#     target = rsp.CellSampling(x, y, headvalues, grid)
#     inverse = rsp.InverseProblem(
#         groundwatermodel=gwf,
#         target=target,
#         regularization_weight= REG_WEIGHT,
#         maxiter=100,
#         relax=0.0,
#     )
#     inverse.formulate()
#     inverse.nonlinear_solve()
#     inverseheads.append(inverse.head)
#     inversehead_i = finalhead.copy(data=inverse.head.reshape(finalhead.shape))
#     error_i = abs(inversehead_i - finalhead).mean().values
#     errors.append(error_i)
#     print(f"Run {i+1}: mean absolute error = {error_i:.4f} m")



# #Print all errors
# # for i, e in enumerate(errors):
# #     print(f"Run {i+1}: {e:.4f} m")

# # Plot a specific run, e.g. run 3
# run_to_inspect = 3
# inversehead_inspect = finalhead.copy(data=inverseheads[run_to_inspect - 1].reshape(finalhead.shape))
# inversehead_inspect.plot.contour(levels=30)

# #Mean error calculation for averaged graph
# mean_inversehead = finalhead.copy(data=np.mean(inverseheads, axis=0).reshape(finalhead.shape))
# mean_inversehead.plot.contour(levels=30)
# plt.figure
# mean_error = mean_inversehead - finalhead
# mean_error.plot.imshow(levels=np.arange(-1.0, 1.0, 0.1))
# print("Mean error:", abs(mean_error).mean().values)

# # %%

# #Plotting the error variability per run
# plt.bar(range(1, n_runs + 1), errors)
# plt.xlabel("Run")
# plt.ylabel("Mean absolute error (m)")
# plt.title(f"Error per run | N_PIEZOMETERS={N_PIEZOMETERS} | T={TRANSMISSIVITY} | reg_weight={REG_WEIGHT}")
# plt.show()

# %%

import datetime
print(f"End of run at {datetime.datetime.now().strftime('%H:%M:%S')}")