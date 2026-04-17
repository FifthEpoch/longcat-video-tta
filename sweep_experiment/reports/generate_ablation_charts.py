"""Generate ablation study charts for the AdaSteer feature analysis slides."""
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
DARK_BLUE = "#1E40AF"
LIGHT_BLUE = "#93C5FD"
RED = "#DC2626"
LIGHT_RED = "#FCA5A5"
GREEN = "#16A34A"
ORANGE = "#EA580C"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"

# ═══════════════════════════════════════════════════════════════════════
# Data: AdaSteer Ablation (Panda-70M, N=100, G=4.0, 10 steps, lr=0.005)
#
# Source: panda_adasteer_ablation config results from cluster.
#   - "Bare" = AS_BARE (no ES, no CLIP, no augmentation)
#   - "+ ES"  = AS_ES1  (es_check_every=1, es_patience=2)
#   - Step/LR sweep data from panda_adasteer_steps_lr (with augmentation)
# ═══════════════════════════════════════════════════════════════════════

ablation = {
    "No-TTA\nBaseline": {
        "fvd": 641.1,
        "train_net": 0, "es_check": 0, "gen": 80.4, "total": 80.4,
        "label": "No adaptation",
    },
    "Bare\nAdaSteer": {
        "fvd": 561.1,
        "train_net": 54, "es_check": 0, "gen": 80, "total": 134,
        "label": "10 steps, lr=0.005",
    },
    "+ Early\nStopping": {
        "fvd": 556.1,
        "train_net": 54, "es_check": 48, "gen": 80, "total": 182,
        "label": "es_check_every=1, patience=2",
    },
}

keys = list(ablation.keys())
n = len(keys)
baseline_fvd = ablation["No-TTA\nBaseline"]["fvd"]


# ═══════════════════════════════════════════════════════════════════════
# CHART 1: Two-Panel — FVD Improvement (left) + Time Cost (right)
# ═══════════════════════════════════════════════════════════════════════

fig, (ax_fvd, ax_time) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.35})

x = np.arange(n)
w = 0.55

# --- Left panel: FVD ---
fvd_vals = [ablation[k]["fvd"] for k in keys]
delta_fvd = [v - baseline_fvd for v in fvd_vals]
bar_colors = [GRAY, DARK_BLUE, "#3B82F6"]

bars = ax_fvd.bar(x, fvd_vals, w, color=bar_colors, edgecolor="white", linewidth=1.5, alpha=0.9)

ax_fvd.axhline(y=baseline_fvd, color=GRAY, linestyle="--", alpha=0.4, linewidth=1.5)

for xi, (bar, fvd, dfvd) in enumerate(zip(bars, fvd_vals, delta_fvd)):
    if dfvd == 0:
        label_text = f"{fvd:.0f}"
        color = GRAY
    else:
        label_text = f"{fvd:.0f}\n({dfvd:+.0f})"
        color = GREEN if dfvd < 0 else RED
    ax_fvd.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                label_text, ha="center", va="bottom", fontsize=13, fontweight="bold", color=color)

ax_fvd.set_xticks(x)
ax_fvd.set_xticklabels(keys, fontsize=12)
ax_fvd.set_ylabel("FVD (lower = better)")
ax_fvd.set_title("FVD by Configuration", fontsize=16, fontweight="bold")
ax_fvd.set_ylim(450, 700)

# Marginal gain annotation
ax_fvd.annotate(
    "Only 5 FVD\nbetter than Bare",
    xy=(2, 556), xytext=(2.35, 490),
    fontsize=11, color="#3B82F6", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=1.5), ha="left")


# --- Right panel: Training Time ---
train_net = [ablation[k]["train_net"] for k in keys]
es_check = [ablation[k]["es_check"] for k in keys]
gen_time = [ablation[k]["gen"] for k in keys]

b1 = ax_time.bar(x, gen_time, w, label="Generation (fixed cost)",
                 color=LIGHT_GRAY, edgecolor="white", linewidth=1.5)
b2 = ax_time.bar(x, train_net, w, bottom=gen_time, label="TTA Training",
                 color=DARK_BLUE, edgecolor="white", linewidth=1.5, alpha=0.85)
b3 = ax_time.bar(x, es_check, w, bottom=[g + t for g, t in zip(gen_time, train_net)],
                 label="ES Validation Overhead", color=RED, edgecolor="white", linewidth=1.5, alpha=0.85)

totals = [g + t + e for g, t, e in zip(gen_time, train_net, es_check)]
for xi, total in enumerate(totals):
    ax_time.text(xi, total + 2, f"{total:.0f}s", ha="center", va="bottom",
                 fontsize=13, fontweight="bold")

# ES overhead annotation
ax_time.annotate(
    f"+48s ES overhead\n(89% of training time)\nfor only 5 FVD gain",
    xy=(2, gen_time[2] + train_net[2] + es_check[2] / 2),
    xytext=(2.45, gen_time[2] + 25),
    fontsize=10, color=RED, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5), ha="left")

ax_time.set_xticks(x)
ax_time.set_xticklabels(keys, fontsize=12)
ax_time.set_ylabel("Time per Video (seconds)")
ax_time.set_title("Training Time Breakdown", fontsize=16, fontweight="bold")
ax_time.legend(loc="upper left", framealpha=0.9, fontsize=11)
ax_time.set_ylim(0, 210)

fig.suptitle("AdaSteer Feature Ablation: FVD Improvement vs Cost",
             fontsize=18, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ablation_fvd_time.png"))
plt.close(fig)
print("  [1/2] ablation_fvd_time.png")


# ═══════════════════════════════════════════════════════════════════════
# CHART 2: Early Stopping Mechanism — step-by-step timeline
#
# Visualizes what happens during a 10-step training run with
# es_check_every=1 and es_patience=2.
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.set_xlim(-0.5, 12)
ax.set_ylim(-1.5, 5.5)
ax.set_axis_off()

ax.text(5.75, 5.2, "Early Stopping with check_every=1, patience=2 (10 training steps)",
        ha="center", fontsize=16, fontweight="bold", color="#1F2937")

# Draw timeline
y_train = 3.8
y_es = 2.0
y_time = 0.5

# Step boxes
step_w = 0.85
for i in range(10):
    x_pos = 0.5 + i * 1.1
    rect = mpatches.FancyBboxPatch(
        (x_pos, y_train - 0.3), step_w, 0.6,
        boxstyle="round,pad=0.08", facecolor="#DBEAFE", edgecolor=DARK_BLUE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x_pos + step_w / 2, y_train, f"Step {i+1}",
            ha="center", va="center", fontsize=9, fontweight="bold", color=DARK_BLUE)

# ES check boxes (after each step except step 1)
for i in range(1, 10):
    x_pos = 0.5 + i * 1.1
    rect = mpatches.FancyBboxPatch(
        (x_pos, y_es - 0.3), step_w, 0.6,
        boxstyle="round,pad=0.08", facecolor="#FEE2E2", edgecolor=RED, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x_pos + step_w / 2, y_es, f"Check",
            ha="center", va="center", fontsize=9, fontweight="bold", color=RED)
    ax.annotate("", xy=(x_pos + step_w / 2, y_es + 0.3),
                xytext=(x_pos + step_w / 2, y_train - 0.3),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0, alpha=0.5))

# Labels
ax.text(-0.3, y_train, "Training\nSteps", ha="center", va="center",
        fontsize=11, fontweight="bold", color=DARK_BLUE)
ax.text(-0.3, y_es, "ES Validation\nChecks", ha="center", va="center",
        fontsize=11, fontweight="bold", color=RED)

# Time annotations
time_train = 54
time_per_check = 48 / 9
total_checks = 9
total_es = 48

ax.text(5.75, y_time + 0.3,
        f"Training:  10 steps x ~5.4s = {time_train}s",
        ha="center", fontsize=13, fontweight="bold", color=DARK_BLUE)
ax.text(5.75, y_time - 0.3,
        f"ES Checks: {total_checks} checks x ~{time_per_check:.1f}s = {total_es}s   "
        f"(each check = 1 DiT forward pass at a random timestep)",
        ha="center", fontsize=12, fontweight="bold", color=RED)
ax.text(5.75, y_time - 0.9,
        f"Total training time: {time_train}s + {total_es}s = {time_train + total_es}s   "
        f"(vs {time_train}s without ES = 1.9x slowdown)",
        ha="center", fontsize=13, fontweight="bold", color="#1F2937")

# Result callout
rect = mpatches.FancyBboxPatch(
    (1.0, -1.3), 9.5, 0.6,
    boxstyle="round,pad=0.1", facecolor="#FEF3C7", edgecolor=ORANGE, linewidth=2)
ax.add_patch(rect)
ax.text(5.75, -1.0,
        "Result: ES never triggers early (best_step = max_steps for most videos) "
        "— all 48s of checking is pure overhead",
        ha="center", va="center", fontsize=11, fontweight="bold", color=ORANGE)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "es_mechanism_timeline.png"))
plt.close(fig)
print("  [2/2] es_mechanism_timeline.png")

print(f"\nAblation charts saved to {OUT}/")
