# make_conductance_scenarios.py
#
# Generates scenario copies of the IBRAHYM boundary-condition datasets with
# scaled conductance values. The original core files are only READ, never
# modified: for each factor a new file is written next to the originals with a
# "-cond{FACTOR}" suffix (e.g. ibrahym-rivers-100m-cond0.5.nc).
#
# Run this from the Sebastiaan/ folder.
#
# Note: overlandflow has no `conductance` variable in its file (its conductance
# is set via constant_conductance in the main script), so it is not handled
# here — scale it manually in the main script instead.

from pathlib import Path

import xarray as xr

# Anchor to this script's own location so it works no matter which directory
# you launch it from (VS Code's "Run" button uses the workspace root, not this
# folder). Sebastiaan/ -> ../../case/ibrahym/
BASE = (Path(__file__).resolve().parent / ".." / ".." / "case" / "ibrahym").resolve()
print(f"reading/writing datasets in: {BASE}")

# Datasets that actually carry a `conductance` variable.
FILES = [
    "ibrahym-rivers-100m.nc",
    "ibrahym-largerivers-100m.nc",
    "ibrahym-drains-100m.nc",
    "ibrahym-tiledrainage-100m.nc",
]

# One scenario set per factor.
FACTORS = [0.5, 2, 3]


def scale_file(filename, factor):
    src = BASE / filename
    if not src.exists():
        raise FileNotFoundError(f"input not found: {src}")
    ds = xr.open_dataset(src)
    if "conductance" not in ds.data_vars:
        raise KeyError(f"{filename} has no 'conductance' variable")

    # Multiplication preserves NaNs; all other variables (stage, elevation,
    # spatial_ref/CRS, ...) are carried through unchanged. Arithmetic drops the
    # variable's .encoding (compression, dtype, chunking), so re-apply it or the
    # file is written uncompressed and balloons in size.
    enc = ds["conductance"].encoding
    ds["conductance"] = ds["conductance"] * factor
    ds["conductance"].encoding = enc

    # "ibrahym-rivers-100m.nc" -> "ibrahym-rivers-100m-cond0.5.nc"
    stem, ext = filename.rsplit(".", 1)
    out = BASE / f"{stem}-cond{factor}.{ext}"
    ds.to_netcdf(out)  # carries the source compression/encoding forward
    ds.close()
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    for factor in FACTORS:
        print(f"factor {factor}:")
        for filename in FILES:
            scale_file(filename, factor)
    print("done.")
