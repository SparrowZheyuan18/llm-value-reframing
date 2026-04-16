import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("data/attack_results.json") as f:
    data = json.load(f)

MODELS = [
    ("deepseek-r1",     "DeepSeek R1",       "Chinese"),
    ("deepseek-v3.2",   "DeepSeek V3.2",     "Chinese"),
    ("llama3-70b",      "Llama 3.3 70B",     "Western"),
    ("llama4-maverick", "Llama 4 Maverick",  "Western"),
    ("mistral-large-3", "Mistral Large 3",   "European"),
    ("qwen3-32b",       "Qwen3 32B",         "Chinese"),
    ("qwen3-235b",      "Qwen3 235B",        "Chinese"),
]

ORIGIN_COLOR = {
    "Chinese":  "#4E8EBD",
    "Western":  "#E07B39",
    "European": "#6BAA75",
}

CONDITION_LABELS = ["Direct (d=0)", "Adjacent (d=1)", "Opposing (d=2)"]
CONDITION_HATCHES = ["", "//", "xx"]

def compute_asr(entries):
    stats = {}
    for dr in [0, 1, 2]:
        valid = [c["comply"] for e in entries for c in e["conditions"]
                 if c["distance_rank"] == dr and c["comply"] != -1 and c["comply"] is not None]
        asr = 100 * sum(1 for v in valid if v == 1) / len(valid) if valid else 0.0
        stats[dr] = asr
    return stats

model_stats = []
for model_id, label, origin in MODELS:
    entries = [e for e in data if e["model"] == model_id]
    stats = compute_asr(entries)
    model_stats.append((label, origin, stats[0], stats[1], stats[2]))

# ── layout ──────────────────────────────────────────────────────────────────
n_models = len(model_stats)
n_conditions = 3
bar_width = 0.22
group_gap = 0.15
x = np.arange(n_models) * (n_conditions * bar_width + group_gap)

fig, ax = plt.subplots(figsize=(13, 6))

for ci, (label, hatch) in enumerate(zip(CONDITION_LABELS, CONDITION_HATCHES)):
    for mi, (m_label, origin, d0, d1, d2) in enumerate(model_stats):
        vals = [d0, d1, d2]
        color = ORIGIN_COLOR[origin]
        xpos = x[mi] + ci * bar_width
        ax.bar(xpos, vals[ci], width=bar_width,
               color=color, alpha=0.85 if ci == 0 else (0.65 if ci == 1 else 0.45),
               hatch=hatch, edgecolor="white", linewidth=0.6)
        ax.text(xpos, vals[ci] + 0.4, f"{vals[ci]:.1f}",
                ha="center", va="bottom", fontsize=7, color="#333333")

# x-axis ticks at group center
group_center = x + bar_width  # center of 3 bars
ax.set_xticks(group_center)
ax.set_xticklabels([m[0] for m in model_stats], fontsize=10)

ax.set_ylabel("Attack Success Rate (%)", fontsize=11)
ax.set_title("ASR by Model and Value Distance\n(errors excluded; Mistral Pixtral omitted)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 40)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── legends ──────────────────────────────────────────────────────────────────
# Condition legend (hatch / alpha)
cond_patches = [
    mpatches.Patch(facecolor="#888888", alpha=0.85, hatch="",   label="Direct (d=0)"),
    mpatches.Patch(facecolor="#888888", alpha=0.65, hatch="//", label="Adjacent (d=1)"),
    mpatches.Patch(facecolor="#888888", alpha=0.45, hatch="xx", label="Opposing (d=2)"),
]
# Origin legend (color)
origin_patches = [
    mpatches.Patch(facecolor=ORIGIN_COLOR["Chinese"],  label="Chinese"),
    mpatches.Patch(facecolor=ORIGIN_COLOR["Western"],  label="Western"),
    mpatches.Patch(facecolor=ORIGIN_COLOR["European"], label="European"),
]

leg1 = ax.legend(handles=cond_patches,   loc="upper right", fontsize=9, title="Condition",      framealpha=0.9)
leg2 = ax.legend(handles=origin_patches, loc="upper left",  fontsize=9, title="Model Origin",   framealpha=0.9)
ax.add_artist(leg1)

plt.tight_layout()
plt.savefig("analysis/asr_bar_chart.png", dpi=150, bbox_inches="tight")
print("Saved: analysis/asr_bar_chart.png")
plt.show()
