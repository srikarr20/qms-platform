import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

real1 = json.loads(
    (ROOT / "experiments/qms_real_001/evidence/qms_real_001_results.json").read_text()
)
real3 = json.loads(
    (ROOT / "experiments/qms_real_003/evidence/qms_real_003_results.json").read_text()
)
real5 = json.loads(
    (ROOT / "experiments/qms_real_005/evidence/qms_real_005_health_score.json").read_text()
)

freq = [r["frequency_hz"] for r in real1["conditions"]]
qnd = [r["QNDFid"] for r in real1["conditions"]]

ed = [r["effective_dimension"] for r in real3["conditions"]]
pc1 = [r["first_pc_fraction"] for r in real3["conditions"]]
radial_std = [r["radial_std"] for r in real3["conditions"]]
degradation = [r["degradation_index"] for r in real5["conditions"]]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

# Panel A: QND fidelity and effective dimension vs operating condition
ax = axes[0]
ax2 = ax.twinx()

ax.plot(freq, qnd, marker="o", label="QND fidelity")
ax2.plot(freq, ed, marker="s", label="Effective dimension")

ax.set_xlabel("Operating frequency (Hz)")
ax.set_ylabel("QND fidelity")
ax2.set_ylabel("Effective dimension")
ax.set_title("(a) Experimental degradation trajectory")
ax.grid(alpha=0.25)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8)

# Panel B: label-free representation metrics vs QND fidelity
ax = axes[1]
ax.plot(qnd, ed, marker="o", label="Effective dimension")
ax.plot(qnd, pc1, marker="s", label="First-PC fraction")

# radial_std has very different units; normalize only for visual comparison
rmin, rmax = min(radial_std), max(radial_std)
radial_norm = [(x - rmin) / (rmax - rmin) for x in radial_std]
ax.plot(qnd, radial_norm, marker="^", label="Radial std (normalized)")

ax.set_xlabel("QND fidelity")
ax.set_ylabel("Representation metric")
ax.set_title("(b) Label-free diagnostics track QND")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

# Panel C: provisional degradation index
ax = axes[2]
ax.plot(qnd, degradation, marker="o")
ax.set_xlabel("QND fidelity")
ax.set_ylabel("Measurement Degradation Index")
ax.set_title("(c) Reference-relative degradation index")
ax.grid(alpha=0.25)

fig.suptitle(
    "Label-free IQ representation diagnostics track experimental QND degradation",
    fontsize=12
)
fig.tight_layout()

out = Path(__file__).with_name("figure2_iq_diagnostics.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(out)
