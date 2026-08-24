"""
Visualising minimum-curvature regularization via a sinusoidal recharge field.

Minimum curvature penalizes the squared second derivative (curvature) of the
recharge field: J_reg = (w/2) || L_r r ||^2, with L_r the Laplacian. A sinusoid
is the natural probe because the Laplacian scales it by its squared wavenumber.

Using the code's own convention (MinimumCurvature.from_sinusoid), the reference
field is
        r(x) = A * sin(pi * x / R)
    A : peak recharge anomaly        [m/day]   -- how large the anomalies are
    R : half-wavelength              [m]       -- the length scale of variation

Its curvature is r''(x) = -A (pi/R)^2 sin(pi x / R), peak magnitude A (pi/R)^2.
So the minimum-curvature PENALTY energy per unit length is

        P(R) proportional to A^2 (pi/R)^4   ~   1 / R^4

i.e. halving the feature size R multiplies the penalty ~16x. Minimum curvature
is therefore a LOW-PASS FILTER: broad (large-R) recharge structure passes almost
free, fine (small-R) structure is strongly suppressed. The weight w sets where
that cutoff sits. Expressing it via (A, R) is the physical, grid-independent
reparameterization the Matern objective adopts.
"""

import matplotlib.pyplot as plt
import numpy as np

# --- parameters (edit freely) ---
A = 1.0                       # amplitude (peak recharge anomaly), arbitrary units
R0 = 1.0                      # reference half-wavelength for the labelled panel
R_OVERLAY = [2.0, 1.0, 0.5]   # a broad, medium, fine length scale (same A)
DOMAIN = 4.0                  # x-range shown in the overlay panel

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

# ---------------------------------------------------------------------------
# (a) the labelled conceptual field: r(x) = A sin(pi x / R)
# ---------------------------------------------------------------------------
ax = axes[0]
x = np.linspace(0, 2 * R0, 400)
r = A * np.sin(np.pi * x / R0)
ax.plot(x, r, color="#1f77b4", lw=2)
ax.axhline(0, color="gray", lw=0.8)

# amplitude A: vertical arrow at the first peak (x = R0/2)
ax.annotate("", xy=(R0 / 2, A), xytext=(R0 / 2, 0),
            arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.5))
ax.text(R0 / 2 + 0.05 * R0, A / 2, "A\n(peak anomaly,\nm/day)",
        color="#d62728", va="center", fontsize=9)

# half-wavelength R: horizontal span between consecutive zeros (0 -> R0)
ax.annotate("", xy=(R0, -0.28 * A), xytext=(0, -0.28 * A),
            arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.5))
ax.text(R0 / 2, -0.42 * A, "R  (half-wavelength,\nlength scale of variation, m)",
        color="#2ca02c", ha="center", va="top", fontsize=9)

ax.set_title(r"Conceptual recharge field  $r(x)=A\,\sin(\pi x / R)$")
ax.set_xlabel("distance x")
ax.set_ylabel("recharge anomaly")
ax.set_ylim(-0.75 * A, 1.25 * A)
ax.set_xticks([])

# ---------------------------------------------------------------------------
# (b) same amplitude, different length scales: broad -> fine
# ---------------------------------------------------------------------------
ax = axes[1]
xx = np.linspace(0, DOMAIN, 800)
colors = ["#2ca02c", "#1f77b4", "#d62728"]
for R, c in zip(R_OVERLAY, colors):
    ax.plot(xx, A * np.sin(np.pi * xx / R), color=c, lw=1.8,
            label=f"R = {R:g}  ({'broad' if R>=2 else 'fine' if R<=0.5 else 'medium'})")
ax.axhline(0, color="gray", lw=0.8)
ax.set_title("Same amplitude A, different length scale R")
ax.set_xlabel("distance x")
ax.set_ylabel("recharge anomaly")
ax.legend(fontsize=8, loc="upper right")
ax.set_xticks([])

# ---------------------------------------------------------------------------
# (c) penalty vs length scale: the low-pass character
# ---------------------------------------------------------------------------
ax = axes[2]
R = np.geomspace(0.2, 5.0, 200)
penalty = (np.pi / R) ** 4          # ~ curvature energy per unit length (A=1)
ax.loglog(R, penalty, color="#9467bd", lw=2)
ax.set_title(r"Minimum-curvature penalty  $\propto (\pi/R)^4 \sim R^{-4}$")
ax.set_xlabel("length scale R")
ax.set_ylabel("penalty (relative)")

# annotate the two regimes
ax.annotate("fine features:\nhuge penalty\n(suppressed)",
            xy=(0.3, (np.pi / 0.3) ** 4), xytext=(0.3, 3e4),
            fontsize=8, color="#d62728", ha="left",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.annotate("broad features:\nnegligible penalty\n(pass through)",
            xy=(4.0, (np.pi / 4.0) ** 4), xytext=(1.1, 5e-2),
            fontsize=8, color="#2ca02c", ha="left",
            arrowprops=dict(arrowstyle="->", color="#2ca02c"))

fig.suptitle("Minimum-curvature regularization as a low-pass filter on recharge",
             fontsize=13)
fig.savefig("minimum_curvature_sinusoid.png", dpi=200, bbox_inches="tight")
fig.savefig("minimum_curvature_sinusoid.pdf", bbox_inches="tight")
print("saved minimum_curvature_sinusoid.png / .pdf")
