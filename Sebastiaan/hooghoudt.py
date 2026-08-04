# hooghoudt_average_head.py
# Hooghoudt drainage: water-table profile between two drains, and its
# average head over the parcel. Linearized case (D >> h), homogeneous k.
#
# Steady state: drain discharge q = recharge R.
# Profile between drains (x = 0 and x = L, drains at both ends):
#     h(x) = (R / (2*k*D)) * x * (L - x)
# By symmetry the mound is symmetric about x = L/2, so we build half (0..L/2).

# %%
import numpy as np
import matplotlib.pyplot as plt

# ---- Worked-example inputs (round numbers) ----
R = 0.001        # recharge [m/day]   (1 mm/day)
k = 1.0          # hydraulic conductivity [m/day]
D = 10.0         # below-drain flow-layer thickness [m]   (need D >> h)
L = 100.0        # drain spacing [m]

kD = k * D       # transmissivity [m^2/day]  -- this is your single kD

# ---- Head profile h(x) ----
# full domain 0..L, but we'll plot the half 0..L/2 (symmetry)
x_half = np.linspace(0.0, L / 2.0, 200)
h_half = (R / (2.0 * kD)) * x_half * (L - x_half)

h_max = h_half[-1]   # at x = L/2, the midpoint mound height
print(f"Max mound height h(L/2) = {h_max:.4f} m")
print(f"Check D >> h:  D = {D} m,  h_max = {h_max:.4f} m  -> ratio D/h = {D/h_max:.1f}")

# ---- Average head over the parcel ----
# Analytical: mean of (R/2kD) x (L-x) over [0, L/2].
# Integral of x(L-x) dx from 0 to L/2  =  L^3/12.  Divide by (L/2): => L^2/6.
# So  h_avg = (R/(2kD)) * (L^2 / 6)  =  R L^2 / (12 kD).
h_avg_analytical = R * L**2 / (12.0 * kD)

# Numerical check (integrate the half-profile, divide by its length)
h_avg_numerical = np.trapz(h_half, x_half) / (L / 2.0)

print(f"\nAverage head (analytical) = {h_avg_analytical:.4f} m")
print(f"Average head (numerical)  = {h_avg_numerical:.4f} m")
print(f"(should match closely)")

# %%
# ---- Plot the Hooghoudt half-profile ----
plt.figure(figsize=(8, 6))
plt.plot(x_half, h_half, lw=2, label="water table h(x)")
plt.axhline(h_avg_analytical, color="red", ls="--",
            label=f"average head = {h_avg_analytical:.3f} m")
plt.axvline(L / 2.0, color="grey", ls=":", alpha=0.6, label="midpoint (L/2)")
plt.scatter([0], [0], color="black", zorder=5)   # drain at x=0
plt.annotate("drain", (0, 0), textcoords="offset points", xytext=(5, 8))
plt.xlabel("Distance from drain, x [m]")
plt.ylabel("Head above drain level [m]")
plt.title(f"Hooghoudt water table (half-profile)\nR={R} m/d, kD={kD:.0f} m²/d, L={L} m")
plt.legend()
plt.tight_layout()
plt.show()