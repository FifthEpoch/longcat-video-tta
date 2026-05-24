"""Generate core presentation charts for Delta Vector PI update."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

BLUE = "#2563EB"
RED = "#DC2626"
GREEN = "#16A34A"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
GRAY = "#6B7280"

# ----------------------------------------------------------------
# Data: Delta Vector Step x LR sweep (Panda-70M, N=100)
# ----------------------------------------------------------------

base_psnr = 18.612
base_fvd = 641.1

steps_list = [5, 10, 15, 20]
lr_list = [0.001, 0.005, 0.01, 0.05]

data = {
    (5,  0.001): (18.576, 634.1),
    (5,  0.005): (18.603, 571.2),
    (5,  0.01):  (18.611, 589.1),
    (5,  0.05):  (18.549, 591.4),
    (10, 0.001): (18.568, 633.9),
    (10, 0.005): (18.590, 568.7),
    (10, 0.01):  (18.589, 568.4),
    (10, 0.05):  (17.587, 605.7),
    (15, 0.001): (18.592, 649.9),
    (15, 0.005): (18.560, 588.9),
    (15, 0.01):  (18.443, 629.5),
    (15, 0.05):  (np.nan, np.nan),
    (20, 0.001): (18.498, 637.4),
    (20, 0.005): (18.488, 657.6),
    (20, 0.01):  (18.586, 608.5),
    (20, 0.05):  (np.nan, np.nan),
}

dfvd = np.full((len(steps_list), len(lr_list)), np.nan)
dpsnr = np.full((len(steps_list), len(lr_list)), np.nan)

for i, s in enumerate(steps_list):
    for j, lr in enumerate(lr_list):
        if (s, lr) in data and not np.isnan(data[(s, lr)][0]):
            dpsnr[i, j] = data[(s, lr)][0] - base_psnr
            dfvd[i, j] = data[(s, lr)][1] - base_fvd

fig, ax = plt.subplots(figsize=(8, 5))

masked = np.ma.masked_invalid(dfvd)
cmap = plt.cm.RdYlGn_r
im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=-80, vmax=20)

ax.set_xticks(range(len(lr_list)))
ax.set_xticklabels([str(lr) for lr in lr_list])
ax.set_yticks(range(len(steps_list)))
ax.set_yticklabels([str(s) for s in steps_list])
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Optimization Steps")
ax.set_title("Delta Vector Sweep: FVD Change from Baseline")

for i in range(len(steps_list)):
    for j in range(len(lr_list)):
        if not np.isnan(dfvd[i, j]):
            fvd_txt = f"{dfvd[i,j]:+.0f}"
            psnr_txt = f"({dpsnr[i,j]:+.2f} dB)"
            color = "white" if abs(dfvd[i, j]) > 40 else "black"
            ax.text(j, i - 0.12, fvd_txt, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)
            ax.text(j, i + 0.22, psnr_txt, ha="center", va="center",
                    fontsize=9, color=color, alpha=0.85)
        else:
            ax.text(j, i, "N/A", ha="center", va="center",
                    fontsize=11, color=GRAY)

cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="delta-FVD (green = better)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "step_lr_heatmap.png"))
plt.close(fig)
print("  [1/3] step_lr_heatmap.png")

# ----------------------------------------------------------------
# CHART 2: Parameter Efficiency Log-Scale
# ----------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

methods_pe = [
    "Full Model\n(14B)",
    "LoRA\n(rank=1)",
    "Delta Vector\n(Delta-B, dim=128)",
    "Delta Vector\n(Delta-A)",
]
params_pe = [14e9, 1000, 128, 1]
fvd_delta_pe = [0, 12.5, -69.6, -72.4]
markers = ["X", "D", "s", "o"]
colors_pe = ["#FCA5A5", "#FCD34D", "#93C5FD", "#4ADE80"]
sizes = [200, 150, 150, 250]

for m, p, f, marker, c, s in zip(
    methods_pe, params_pe, fvd_delta_pe, markers, colors_pe, sizes
):
    ax.scatter(
        p, f, s=s, c=c, marker=marker, edgecolors="black",
        linewidths=1.5, zorder=5, label=m,
    )

ax.set_xscale("log")
ax.set_xlabel("Trainable Parameters (log scale)")
ax.set_ylabel("delta-FVD (lower = better)")
ax.set_title("Parameter Efficiency: Fewer Params, Better Temporal Coherence")
ax.axhline(y=0, color=GRAY, linestyle="--", alpha=0.5, label="Baseline (No-TTA)")
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

ax.fill_between([0.5, 200], [-80, -80], [0, 0], alpha=0.06, color=GREEN)
ax.text(5, -75, "Better than\nbaseline", fontsize=10, color=GREEN, alpha=0.7,
        ha="center")
ax.fill_between([0.5, 2e10], [0, 0], [20, 20], alpha=0.06, color=RED)
ax.text(1e6, 15, "Worse than baseline", fontsize=10, color=RED, alpha=0.7,
        ha="center")

ax.set_xlim(0.5, 2e10)
ax.set_ylim(-85, 25)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "param_efficiency.png"))
plt.close(fig)
print("  [2/3] param_efficiency.png")

# ----------------------------------------------------------------
# CHART 3: Architecture Diagram
# ----------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.set_axis_off()


def draw_box(ax, x, y, w, h, text, color="#E5E7EB", text_color="black",
             fontsize=11, bold=False):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="#374151", linewidth=1.5,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, color=text_color, fontweight=weight,
    )


def draw_arrow(ax, x1, y1, x2, y2, color="#374151"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
    )


ax.text(
    5, 6.6, "Delta Vector: Delta Injection into DiT AdaLN",
    fontsize=16, ha="center", fontweight="bold",
)

draw_box(ax, 0.3, 4.8, 2.0, 0.8, "Noisy Latent\nz_t", color="#DBEAFE")
draw_box(ax, 0.3, 3.2, 2.0, 0.8, "Timestep\nEmbedding", color="#FEF3C7")
draw_box(ax, 3.3, 3.2, 1.8, 0.8, "+ delta", color="#DCFCE7",
         text_color=GREEN, fontsize=18, bold=True)
ax.text(4.2, 2.5, "<-- 1 learnable\n     scalar", fontsize=10,
        color=GREEN, fontweight="bold")
draw_box(ax, 5.8, 3.2, 2.2, 0.8, "AdaLN\n(scale & shift)", color="#F3E8FF")
draw_box(ax, 5.8, 4.8, 2.2, 0.8, "DiT Transformer\nBlock (x48)", color="#DBEAFE")
draw_box(ax, 5.8, 1.3, 2.2, 0.8, "Predicted\nVelocity v_theta", color="#DBEAFE")

draw_arrow(ax, 2.3, 5.2, 5.8, 5.2)
draw_arrow(ax, 2.3, 3.6, 3.3, 3.6)
draw_arrow(ax, 5.1, 3.6, 5.8, 3.6)
draw_arrow(ax, 6.9, 4.8, 6.9, 4.0)
draw_arrow(ax, 6.9, 3.2, 6.9, 2.1)

draw_box(ax, 3.0, 0.3, 3.0, 0.8, "Flow-Matching Loss\non Conditioning Frames",
         color="#FEE2E2")
draw_arrow(ax, 6.9, 1.3, 5.5, 0.7)

ax.text(
    1.3, 0.7, "TRAIN:\nOptimize delta", fontsize=10, color=RED,
    fontweight="bold", ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", alpha=0.7),
)
ax.text(
    9.0, 5.2, "INFER:\nApply delta to\ngenerate\nfuture frames", fontsize=10,
    color=BLUE, fontweight="bold", ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#DBEAFE", alpha=0.7),
)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "architecture_diagram.png"))
plt.close(fig)
print("  [3/3] architecture_diagram.png")

print(f"\nCore charts saved to {OUT}/")
