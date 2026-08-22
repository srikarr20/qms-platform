import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

files = {
    "rolling": ROOT / "experiments/qms_real_012/evidence/qms_real_012_causal_drift.json",
    "frozen": ROOT / "experiments/qms_real_013/evidence/qms_real_013_frozen_reference.json",
    "auto": ROOT / "experiments/qms_real_014/evidence/qms_real_014_auto_baseline_lock.json",
    "persistent": ROOT / "experiments/qms_real_015/evidence/qms_real_015_persistent_drift.json",
    "slope": ROOT / "experiments/qms_real_016/evidence/qms_real_016_drift_acceleration.json",
}

d = {k: json.loads(p.read_text()) for k, p in files.items()}

labels = [
    "Rolling\nreference",
    "Frozen early\nreference",
    "Auto baseline\nlock",
    "Persistent\ndrift",
    "Slope\nacceleration",
]

first_warning = [
    None,
    d["frozen"]["summary"]["first_warning"]["frame_start"],
    None,
    d["persistent"]["summary"]["first_persistent_warning"]["frame_start"],
    None,
]

first_alert = [
    None,
    d["frozen"]["summary"]["first_drift_alert"]["frame_start"],
    d["auto"]["summary"]["first_drift_alert"]["frame_start"],
    d["persistent"]["summary"]["first_persistent_drift_alert"]["frame_start"],
    None,
]

retrospective_reference = 3034

x = list(range(len(labels)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: first warning / alert timing
ax = axes[0]

for i, value in enumerate(first_warning):
    if value is not None:
        ax.scatter(i - 0.08, value, marker="o", s=60, label="First warning" if i == 1 else None)

for i, value in enumerate(first_alert):
    if value is not None:
        ax.scatter(i + 0.08, value, marker="s", s=60, label="First alert" if i == 1 else None)

ax.axhline(
    retrospective_reference,
    linestyle="--",
    linewidth=1.2,
    label="Retrospective drift region (~3034)"
)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Frame start")
ax.set_title("(a) Causal detection timing")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

# Panel B: qualitative outcome
ax = axes[1]

outcomes = [
    0,   # no trigger
    -1,  # premature
    -1,  # premature
    -1,  # still earlier than retrospective region
    0,   # no trigger
]

ax.bar(x, outcomes)
ax.axhline(0, linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yticks([-1, 0])
ax.set_yticklabels(["Premature trigger", "No trigger"])
ax.set_ylim(-1.4, 0.5)
ax.set_title("(b) Tested prospective outcomes")

fig.suptitle(
    "Tested causal formulations do not establish prospective early warning",
    fontsize=12
)

fig.tight_layout()

out = Path(__file__).with_name("figure5_causal_negative.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(out)
