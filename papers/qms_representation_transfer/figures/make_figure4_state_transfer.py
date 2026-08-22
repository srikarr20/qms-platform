import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

summary = json.loads(
    (ROOT / "experiments/qms_real_023/evidence/qms_real_023_four_dataset_summary.json").read_text()
)

datasets = summary["summary"]["datasets"]

names = [
    "Heralded\nDiffraction",
    "Ghost\nDiffraction",
    "Heralded\nImaging",
    "Ghost\nImaging",
]

drift_pos = [d["normalized_drift_position"] for d in datasets]
max_cos_pos = [d["maximum_cosine_relative_position"] for d in datasets]
min_js_pos = [d["minimum_js_relative_position"] for d in datasets]

initial = [d["state_counts"].get("initial", 0) for d in datasets]
converging = [d["state_counts"].get("converging", 0) for d in datasets]
stable = [d["state_counts"].get("stable_or_mixed", 0) for d in datasets]
drifting = [d["state_counts"].get("drifting", 0) for d in datasets]

x = list(range(len(names)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: normalized positions
ax = axes[0]
ax.plot(x, max_cos_pos, marker="o", label="Maximum cosine")
ax.plot(x, min_js_pos, marker="s", label="Minimum JS")
ax.plot(x, drift_pos, marker="^", label="First detected drift")

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1)
ax.set_ylabel("Normalized acquisition position")
ax.set_title("(a) Transfer of acquisition-state landmarks")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

# Panel B: state counts
ax = axes[1]

bottom1 = initial
bottom2 = [a + b for a, b in zip(initial, converging)]
bottom3 = [a + b + c for a, b, c in zip(initial, converging, stable)]

ax.bar(x, initial, label="Initial")
ax.bar(x, converging, bottom=initial, label="Converging")
ax.bar(x, stable, bottom=bottom2, label="Stable/mixed")
ax.bar(x, drifting, bottom=bottom3, label="Drifting")

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Number of windows")
ax.set_title("(b) Unchanged operational-state classifier")
ax.legend(frameon=False, fontsize=8)

fig.suptitle(
    "Operational state logic transfers across four Glasgow acquisition conditions",
    fontsize=12
)

fig.tight_layout()

out = Path(__file__).with_name("figure4_state_transfer.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(out)
