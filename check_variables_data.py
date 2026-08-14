import xarray as xr
BASE = r"c:\Users\sebas\Documents\case\ibrahym\ibrahym-"   # <- your real path
for name in ["drains-100m", "gxg", "head-l1-100m", "largerivers-100m",
             "overlandflow-100m", "phreatic-head-100m", "rivers-100m",
             "subsoil-100m", "tiledrainage-100m"]:
    ds = xr.open_dataset(f"{BASE}{name}.nc")
    print(f"\n=== {name} ===")
    print("dims:", dict(ds.sizes))
    for v in ds.data_vars:
        u = ds[v].attrs.get("units", "-")
        print(f"  {v:20s} {u:10s} {ds[v].dims}")
    ds.close()