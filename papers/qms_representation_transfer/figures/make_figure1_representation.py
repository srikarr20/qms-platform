import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

rep1 = json.loads(
    (ROOT / "experiments/qms_rep_001/evidence/qms_rep_001_results.json").read_text()
)
rep2 = json.loads(
    (ROOT / "experiments/qms_rep_002/evidence/qms_rep_002_results.json").read_text()
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# Panel A: generic synthetic representation-noise behavior
noise = [r["noise"] for r in rep1]
ed = [r["effective_dimension"] for r in rep1]
evf2 = [r["variance_explained_first_2"] for r in rep1]
corr = [r["energy_correlation"] for r in rep1]

ax = axes[0]
ax.plot(noise, ed, marker="o", label="Effective dimension")
ax.plot(noise, evf2, marker="s", label="EVF first 2")
ax.plot(noise, corr, marker="^", label="Observable correlation")
ax.set_xlabel("Noise level")
ax.set_ylabel("Metric value")
ax.set_title("(a) Synthetic representation diagnostics")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

# Panel B: noisy quantum-tomography representation behavior
count_scale = [r["count_scale"] for r in rep2]
ed2 = [r["effective_dimension"] for r in rep2]
sim = [r["mean_similarity_to_ideal"] for r in rep2]

ax = axes[1]
ax2 = ax.twinx()

ax.plot(count_scale, ed2, marker="o", label="Effective dimension")
ax2.plot(count_scale, sim, marker="s", label="Similarity to ideal")

ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("Count scale")
ax.set_ylabel("Effective dimension")
ax2.set_ylabel("Mean similarity to ideal")
ax.set_title("(b) Noisy tomography ensemble")
ax.grid(alpha=0.25)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8)

fig.suptitle(
    "Representation complexity is not equivalent to measurement quality",
    fontsize=12
)
fig.tight_layout()

out = Path(__file__).with_name("figure1_representation.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(out)
