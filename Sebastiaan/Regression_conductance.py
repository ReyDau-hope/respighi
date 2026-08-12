# %%
import numpy as np
import matplotlib.pyplot as plt

# ---- parameters ----
R  = 0.001        # recharge [m/day]
kD = 2000.0       # transmissivity [m^2/day]
c  = np.linspace(0.0, 5.0, 50)   # drainage resistance sweep [days]

L_values = [60, 100, 230, 320]   # your measured drain spacings [m]
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(L_values)))

plt.figure(figsize=(9, 6))
for L, col in zip(L_values, colors):
    h_bar = R * (L**2/(12*kD) + c)
    slope, intercept = np.polyfit(c, h_bar, 1)      # slope = R, intercept = aquifer term
    plt.plot(c, h_bar, color=col, lw=2,
             label=f"L = {L} m  (aquifer term = {intercept:.4f} m)")
    plt.axhline(intercept, color=col, ls=":", alpha=0.5)   # the intercept (c=0, ideal drain)

plt.xlabel("Drainage resistance  c  (days)")
plt.ylabel("Average head  h̄  (m)")
plt.title(f"Average head vs drainage resistance, per drain spacing\n"
          f"R = {R} m/d,  kD = {kD:.0f} m²/d   (slope = R for all L)")
plt.legend(title="drain spacing")
plt.tight_layout()
plt.show()

# %%
# --- the decomposition table: where resistance overtakes the aquifer term ---
print(f"{'L (m)':>6} | {'aquifer term (m)':>16} | {'c where resistance = aquifer (d)':>32}")
print("-"*60)
for L in L_values:
    aquifer = R * L**2 / (12*kD)
    c_cross = aquifer / R          # = L^2/(12kD)
    print(f"{L:>6} | {aquifer:>16.5f} | {c_cross:>32.2f}")
# %%
R  = 0.001
kD = 2000.0
L_values = [60, 100, 230, 320]
c = np.linspace(0.0, 5.0, 50)

plt.figure(figsize=(9, 6))

# the universal dimensionless curve
c_star_axis = np.linspace(0, 8, 100)
plt.plot(c_star_axis, 1 + c_star_axis, "k-", lw=2, label="h̄/h_ref = 1 + c*  (universal)")

# where each L's realistic c range (1–5 d) maps onto c*
for L in L_values:
    c_star = 12*kD*c / L**2
    plt.plot(c_star, 1 + c_star, "o", ms=3, alpha=0.6, label=f"L = {L} m")

plt.axvline(1.0, color="red", ls="--", alpha=0.6, label="c* = 1 (crossover)")
plt.xlabel("Dimensionless resistance  c* = 12·kD·c / L²")
plt.ylabel("Dimensionless head  h̄ / h_ref")
plt.title("Collapsed (dimensionless) head–resistance relation")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# where does each L land on c* for the realistic resistance range c = 1–5 days?
print(f"{'L (m)':>6} | {'c* at c=1d':>10} | {'c* at c=5d':>10} | regime")
print("-"*55)
for L in L_values:
    cs_lo = 12*kD*1 / L**2
    cs_hi = 12*kD*5 / L**2
    regime = "resistance-controlled" if cs_lo > 1 else ("mixed" if cs_hi > 1 else "aquifer-controlled")
    print(f"{L:>6} | {cs_lo:>10.2f} | {cs_hi:>10.2f} | {regime}")
# %%
