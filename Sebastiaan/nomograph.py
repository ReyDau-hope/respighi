# %%
# nomograph_dimensionless.py
# Nomograph of the dimensionless average head.
# Relationship (from Hooghoudt + Buckingham):  h*  =  R* ( 1/12 + c* )
#   h*  = h_bar / L        (dimensionless average head)
#   R*  = R L / kD         (dimensionless recharge)
#   c*  = kD c / L^2       (dimensionless drainage resistance)
# For a fixed c*, h* is linear in R* with slope (1/12 + c*):
# each c* is a straight line through the origin, steeper for larger c*.

import numpy as np
import matplotlib.pyplot as plt

# %%
# --- axis ranges ---
R_star = np.linspace(0, 5, 200)          # x-axis: dimensionless recharge
c_star_values = [0.0, 1/12, 0.25, 0.5, 1.0, 2.0]   # family of curves

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(c_star_values)))

plt.figure(figsize=(8, 6))
for cs, col in zip(c_star_values, colors):
    slope = 1/12 + cs
    h_star = slope * R_star
    label = f"c* = {cs:.3f}" if cs != 1/12 else "c* = 1/12 (crossover)"
    plt.plot(R_star, h_star, color=col, lw=2, label=label)

plt.xlabel(r"Dimensionless recharge  $R^{*} = RL/kD$")
plt.ylabel(r"Dimensionless average head  $\bar{h}^{*} = \bar{h}/L$")
plt.title(r"Nomograph:  $\bar{h}^{*} = R^{*}\,(1/12 + c^{*})$")
plt.legend(title="dimensionless resistance", loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("nomograph_dimensionless.png", dpi=200, bbox_inches="tight")
plt.show()
# %%
