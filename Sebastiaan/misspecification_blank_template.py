"""
Design schematic for the sigma-misspecification experiment (a methods figure).

A labelled 3x3 grid. Injected noise (sigma_ext) -- the noise actually ADDED to the
observations -- on the vertical axis; assumed noise (sigma_int) -- the uncertainty
the model is TOLD to expect -- on the horizontal axis. The diagonal is the
correctly-specified case; off-diagonal cells are deliberate misspecification:

    sigma_int < sigma_ext : model told data is cleaner than it is -> over-trusts
                            -> chases noise -> rough recharge
    sigma_int = sigma_ext : correctly specified (honest)
    sigma_int > sigma_ext : model told data is noisier than it is -> under-trusts
                            -> leans on smoothing -> flat recharge

Layout matches the results heatmaps (sigma_ext = 0.10 at top, sigma_int across
the bottom). Saves PNG and PDF (use the PDF in LaTeX -- vector, scales cleanly).
"""
#%%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

SIGMA = [0.10, 0.20, 0.50]          # values on both axes, m (edit or make generic)

C_HONEST = "#bcdcc0"                 # correctly specified
C_OVER   = "#f2c4ad"                 # over-trust  (sigma_int < sigma_ext)
C_UNDER  = "#aec6e8"                 # under-trust (sigma_int > sigma_ext)

n = len(SIGMA)
fig, ax = plt.subplots(figsize=(6.0, 5.4))

for i in range(n):                  # row -> sigma_ext
    for j in range(n):              # col -> sigma_int
        if j == i:
            color, label = C_HONEST, "correctly\nspecified"
        elif j < i:
            color, label = C_OVER, "over-trust"
        else:
            color, label = C_UNDER, "under-trust"
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                               facecolor=color, edgecolor="white", lw=2))
        ax.text(j, i, label, ha="center", va="center",
                fontsize=10, fontweight="bold")

for k in range(n):                  # bold outline on the honest diagonal
    ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                           edgecolor="#2f6b34", lw=2.5))

ax.set_xticks(range(n), [f"{v:.2f}" for v in SIGMA])
ax.set_yticks(range(n), [f"{v:.2f}" for v in SIGMA])
ax.set_xlabel(r"assumed noise  $\sigma_{\mathrm{int}}$  (told to the model)  [m]")
ax.set_ylabel(r"injected noise  $\sigma_{\mathrm{ext}}$  (added to the data)  [m]")
ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(-0.5, n - 0.5)
ax.invert_yaxis()                   # sigma_ext = 0.10 at top, matching results
ax.set_aspect("equal")

handles = [
    Patch(facecolor=C_OVER, edgecolor="none",
          label=r"$\sigma_{int}<\sigma_{ext}$  over-trust"),
    Patch(facecolor=C_HONEST, edgecolor="#2f6b34",
          label=r"$\sigma_{int}=\sigma_{ext}$  correctly specified (honest)"),
    Patch(facecolor=C_UNDER, edgecolor="none",
          label=r"$\sigma_{int}>\sigma_{ext}$  under-trust"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          frameon=False, fontsize=8.5, handlelength=1.2)

ax.set_title("Misspecification design: injected vs. assumed observation noise",
             fontsize=11)
fig.tight_layout()
fig.savefig("misspecification_design.png", dpi=200, bbox_inches="tight")
fig.savefig("misspecification_design.pdf", bbox_inches="tight")
print("saved misspecification_design.png / .pdf")
# %%
