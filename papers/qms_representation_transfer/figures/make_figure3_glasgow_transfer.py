import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

hd = json.loads(
    (ROOT / "experiments/qms_real_007/evidence/qms_real_007_convergence.json").read_text()
)
gd = json.loads(
    (ROOT / "experiments/qms_real_017/evidence/qms_real_017_ghost_diffraction_sm.json").read_text()
)
cc = json.loads(
    (ROOT / "experiments/qms_real_018/evidence/qms_real_018_cross_condition_consistency.json").read_text()
)

hd_w = hd["windows"]
gd_w = gd["windows"]

x_hd = [w["frame_start"] for w in hd_w]
x_gd = [w["frame_start"] for w in gd_w]

hd_cos = [w["cosine_to_final"] for w in hd_w]
gd_cos = [w["cosine_to_final"] for w in gd_w]

hd_js = [w["js_divergence_to_final"] for w in hd_w]
gd_js = [w["js_divergence_to_final"] for w in gd_w]

hd_ed = [w["effective_dimension"] for w in hd_w]
gd_ed = [w["effective_dimension"] for w in gd_w]

hd_pc1 = [w["first_pc_fraction"] for w in hd_w]
gd_pc1 = [w["first_pc_fraction"] for w in gd_w]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Panel A: cosine convergence
ax = axes[0, 0]
ax.plot(x_hd, hd_cos, marker="o", markersize=3, label="Heralded diffraction")
ax.plot(x_gd, gd_cos, marker="s", markersize=3, label="Ghost diffraction")
ax.set_xlabel("Frame start")
ax.set_ylabel("Cosine similarity to mature distribution")
ax.set_title("(a) Distribution convergence: cosine")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

# Panel B: JS convergence
ax = axes[0, 1]
ax.plot(x_hd, hd_js, marker="o", markersize=3, label="Heralded diffraction")
ax.plot(x_gd, gd_js, marker="s", markersize=3, label="Ghost diffraction")
ax.set_xlabel("Frame start")
ax.set_ylabel("Jensen–Shannon divergence")
ax.set_title("(b) Distribution convergence: JS divergence")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

# Panel C: effective-dimension trajectories
ax = axes[1, 0]
ax.plot(x_hd, hd_ed, marker="o", markersize=3, label="Heralded diffraction")
ax.plot(x_gd, gd_ed, marker="s", markersize=3, label="Ghost diffraction")
ax.set_xlabel("Frame start")
ax.set_ylabel("Effective dimension")
ax.set_title(
    "(c) Fine representation geometry\n"
    f"cross-condition r = "
    f"{cc['summary']['metric_comparisons']['effective_dimension']['raw_pearson']:.3f}"
)
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

# Panel D: direct trajectory comparison
ax = axes[1, 1]
ax.scatter(hd_cos, gd_cos, s=20, label="Cosine trajectory")
ax.scatter(hd_js, gd_js, s=20, label="JS trajectory")
ax.set_xlabel("Heralded diffraction metric")
ax.set_ylabel("Ghost diffraction metric")
ax.set_title(
    "(d) Cross-condition trajectory agreement\n"
    f"cosine r = "
    f"{cc['summary']['metric_comparisons']['cosine_to_final']['raw_pearson']:.4f}, "
    f"JS r = "
    f"{cc['summary']['metric_comparisons']['js_divergence_to_final']['raw_pearson']:.4f}"
)
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

fig.suptitle(
    "Distribution-level convergence transfers more strongly than fine representation geometry",
    fontsize=12
)

fig.tight_layout()

out = Path(__file__).with_name("figure3_glasgow_transfer.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(out)
