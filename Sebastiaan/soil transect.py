# %%
# soil_transect.py
# Standalone hydrogeological cross-sections from the IBRAHYM subsoil file.
# Shows the 63-layer stack that Respighi compresses into a single kD (RQ1 illustration).

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# %%
# --- Load subsoil ---
subsoil = xr.open_dataset("../../case/ibrahym/ibrahym-subsoil-100m.nc")

XMIN, XMAX = 185_000.0, 205_000.0
YMIN, YMAX = 350_000.0, 370_000.0
subsoil = subsoil.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))

n_layers = subsoil.sizes["layer"]
colors = plt.cm.tab20(np.linspace(0, 1, 20))
layer_colors = [colors[i % 20] for i in range(n_layers)]


def plot_transect(ds, along, fixed_value, ax, title):
    if along == "x":
        line = ds.sel(y=fixed_value, method="nearest")
        axis = line.x.values
        xlabel = "x (m, RD New) — West → East"
    else:
        line = ds.sel(x=fixed_value, method="nearest")
        axis = line.y.values
        xlabel = "y (m, RD New) — South → North"

    top = line["top"].values
    bottom = line["bottom"].values

    patches, facecolors = [], []
    for li in range(top.shape[0]):
        t, b = top[li], bottom[li]
        valid = ~(np.isnan(t) | np.isnan(b))
        if not valid.any():
            continue
        idx = np.where(valid)[0]
        splits = np.where(np.diff(idx) > 1)[0] + 1
        for seg in np.split(idx, splits):
            if seg.size < 2:
                continue
            xs = axis[seg]
            poly_x = np.concatenate([xs, xs[::-1]])
            poly_y = np.concatenate([t[seg], b[seg][::-1]])
            patches.append(Polygon(np.column_stack([poly_x, poly_y]), closed=True))
            facecolors.append(layer_colors[li])

    pc = PatchCollection(patches, facecolors=facecolors, edgecolors="none", alpha=0.9)
    ax.add_collection(pc)
    ax.set_xlim(axis.min(), axis.max())
    ax.set_ylim(np.nanmin(bottom), np.nanmax(top))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Elevation (m NAP)")
    ax.set_title(title)


# %%
# --- Plot both transects through the middle of the window ---
y_mid = float(subsoil.y.mean())
x_mid = float(subsoil.x.mean())

fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 10))
plot_transect(subsoil, along="x", fixed_value=y_mid, ax=ax0,
              title=f"West–East transect at y = {y_mid:.0f} m  ({n_layers} layers)")
plot_transect(subsoil, along="y", fixed_value=x_mid, ax=ax1,
              title=f"South–North transect at x = {x_mid:.0f} m  ({n_layers} layers)")
plt.tight_layout()
plt.savefig("soil_transect.png", dpi=200, bbox_inches="tight")
plt.show()