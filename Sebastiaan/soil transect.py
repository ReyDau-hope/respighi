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
#plt.savefig(r"C:\Users\sebas\Documents\1Thesis\respighi\Findings\soil_transect.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
# --- Per-layer window-averaged kD ---
thickness = subsoil["top"] - subsoil["bottom"]
kD_layer = subsoil["kh"] * thickness
kD_mean = kD_layer.mean(dim=["x", "y"], skipna=True).values   # one kD per layer
layer_ids = subsoil.layer.values

for li, kd in zip(layer_ids, kD_mean):
    print(f"layer {li:2d}: kD = {kd:8.1f} m²/day")
print(f"\nSum over layers: {np.nansum(kD_mean):.1f} m²/day")


# %%
# --- Transect with kD labels on thick-enough bands ---
def plot_transect_labeled(ds, along, fixed_value, ax, title,
                          min_thickness_for_label=40.0, ylim=None):
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
    x_mid_idx = len(axis) // 2   # place labels at the centre of the transect

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

        # Label if this layer is thick enough at the centre of the transect
        t_mid, b_mid = top[li, x_mid_idx], bottom[li, x_mid_idx]
        if np.isfinite(t_mid) and np.isfinite(b_mid):
            band_thickness = t_mid - b_mid
            if ylim is not None:
                # only consider the label if the band centre is within the zoom
                y_centre = (t_mid + b_mid) / 2
                in_view = ylim[0] <= y_centre <= ylim[1]
            else:
                in_view = True
            if band_thickness >= min_thickness_for_label and in_view:
                ax.text(axis[x_mid_idx], (t_mid + b_mid) / 2,
                        f"{kD_mean[li]:.0f}",
                        ha="center", va="center", fontsize=7,
                        color="black",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6))

    pc = PatchCollection(patches, facecolors=facecolors, edgecolors="white",
                         linewidths=0.2, alpha=0.9)
    ax.add_collection(pc)
    ax.set_xlim(axis.min(), axis.max())
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(np.nanmin(bottom), np.nanmax(top))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Elevation (m NAP)")
    ax.set_title(title)


y_mid = float(subsoil.y.mean())
x_mid = float(subsoil.x.mean())

# Full-depth labelled transects
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(14, 11))
plot_transect_labeled(subsoil, "x", y_mid, ax0,
    f"West–East at y = {y_mid:.0f} m — kD labels (m²/day), thick layers only")
plot_transect_labeled(subsoil, "y", x_mid, ax1,
    f"South–North at x = {x_mid:.0f} m — kD labels (m²/day), thick layers only")
plt.tight_layout()
#plt.savefig(r"C:\Users\sebas\Documents\1Thesis\respighi\Findings\soil_transect_kD_labelled.png", dpi=200, bbox_inches="tight")
plt.show()


# %%
# --- Zoomed top panel: near-surface thin layers, fully labelled ---
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(14, 11))
plot_transect_labeled(subsoil, "x", y_mid, ax0,
    f"West–East (top 250 m) at y = {y_mid:.0f} m — kD labels (m²/day)",
    min_thickness_for_label=5.0, ylim=(-250, 30))
plot_transect_labeled(subsoil, "y", x_mid, ax1,
    f"South–North (top 250 m) at x = {x_mid:.0f} m — kD labels (m²/day)",
    min_thickness_for_label=5.0, ylim=(-250, 30))
plt.tight_layout()
#plt.savefig(r"C:\Users\sebas\Documents\1Thesis\respighi\Findings\soil_transect_kD_topsoil_labelled.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
kD_total = (subsoil["kh"] * (subsoil["top"] - subsoil["bottom"])).sum(dim="layer")
print("per-cell total kD:  min", float(kD_total.min()), " max", float(kD_total.max()), " mean", float(kD_total.mean()))

# %%
# --- Map of total column kD across the study window, min/max marked ---
kD_total = (subsoil["kh"] * (subsoil["top"] - subsoil["bottom"])).sum(dim="layer")

# locate min and max cells
kD_vals = kD_total.values
iy_max, ix_max = np.unravel_index(np.nanargmax(kD_vals), kD_vals.shape)
iy_min, ix_min = np.unravel_index(np.nanargmin(kD_vals), kD_vals.shape)
x_max, y_max = float(kD_total.x[ix_max]), float(kD_total.y[iy_max])
x_min, y_min = float(kD_total.x[ix_min]), float(kD_total.y[iy_min])

fig, ax = plt.subplots(figsize=(10, 9))
im = kD_total.plot.imshow(ax=ax, cmap="viridis", add_colorbar=True,
                          cbar_kwargs={"label": "Total column kD (m²/day)"})

# mark extremes
ax.scatter(x_max, y_max, s=120, marker="*", color="red", edgecolor="black", zorder=5,
           label=f"max ≈ {float(kD_total.max()):.0f}")
ax.scatter(x_min, y_min, s=120, marker="o", color="white", edgecolor="black", zorder=5,
           label=f"min ≈ {float(kD_total.min()):.0f}")

ax.set_title("Total transmissivity (kD) across study area — summed over 63 layers")
ax.set_xlabel("x (m, RD New)")
ax.set_ylabel("y (m, RD New)")
ax.set_aspect("equal")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(r"C:\Users\sebas\Documents\1Thesis\respighi\Findings\kD_total_map.png", dpi=200, bbox_inches="tight")
plt.show()

print(f"max kD {float(kD_total.max()):.0f} at (x={x_max:.0f}, y={y_max:.0f})")
print(f"min kD {float(kD_total.min()):.0f} at (x={x_min:.0f}, y={y_min:.0f})")
# %%
kD_total = (subsoil["kh"] * (subsoil["top"] - subsoil["bottom"])).sum(dim="layer")
print("mean of per-cell total kD:", float(kD_total.mean()))
# %%
