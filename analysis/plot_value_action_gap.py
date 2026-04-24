import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

with open("data/civics_attack_results.json") as f:
    civics = json.load(f)
with open("data/pluriharms_attack_results.json") as f:
    pluriharms = json.load(f)

civics     = [e for e in civics     if e["model"] != "mistral-pixtral"]
pluriharms = [e for e in pluriharms if e["model"] != "mistral-pixtral"]
combined   = civics + pluriharms

MODELS = [
    ("deepseek-r1",     "DeepSeek R1"),
    ("deepseek-v3.2",   "DeepSeek V3.2"),
    ("llama3-70b",      "Llama 3.3 70B"),
    ("llama4-maverick", "Llama 4 Maverick"),
    ("mistral-large-3", "Mistral Large 3"),
    ("qwen3-32b",       "Qwen3 32B"),
    ("qwen3-235b",      "Qwen3 235B"),
]

VALUE_ORDER = [
    "Universalism", "Benevolence", "Conformity", "Tradition", "Security",
    "Power", "Achievement", "Self-Direction", "Stimulation",
]

# --- probe agreement rate per (model, v_s) ---
probe_comply  = defaultdict(int)
probe_total   = defaultdict(int)
for e in combined:
    p = e.get("probe", {})
    c = p.get("comply")
    if c == -1 or c is None:
        continue
    probe_comply[(e["model"], e["v_s"])] += int(c == 1)
    probe_total[(e["model"], e["v_s"])]  += 1

# --- direct jailbreak ASR per (model, v_s) ---
direct_comply = defaultdict(int)
direct_total  = defaultdict(int)
for e in combined:
    vs = e["v_s"]
    for cond in e["conditions"]:
        if cond["condition"] != "direct":
            continue
        c = cond["comply"]
        if c == -1 or c is None:
            continue
        direct_comply[(e["model"], vs)] += int(c == 1)
        direct_total[(e["model"], vs)]  += 1

# --- collect points ---
colors = plt.cm.tab10(np.linspace(0, 0.9, len(MODELS)))

fig, ax = plt.subplots(figsize=(9, 8))

# Shaded zones
t = np.linspace(0, 100, 200)
ax.fill_between(t, 100 - t, 100, color="#f28b82", alpha=0.13, zorder=0)   # gap zone
ax.fill_between(t, 0, 100 - t,  color="#aecbfa", alpha=0.13, zorder=0)    # over-refusal zone

# Zone labels
ax.text(72, 88, "Easy-Complying\n(agree but still comply)", fontsize=19,
        color="#c0392b", ha="center", va="center", style="italic", alpha=0.85)
ax.text(35, 18, "Over-Refusal\n(refuse beyond stated values)", fontsize=19,
        color="#2471a3", ha="center", va="center", style="italic", alpha=0.85)

VALUE_TO_CATEGORY = {
    "Universalism":   "Self-Transcendence",
    "Benevolence":    "Self-Transcendence",
    "Conformity":     "Conservation",
    "Tradition":      "Conservation",
    "Security":       "Conservation",
    "Power":          "Self-Enhancement",
    "Achievement":    "Self-Enhancement",
    "Self-Direction": "Openness to Change",
    "Stimulation":    "Openness to Change",
    "Hedonism":       "Openness to Change",
}

CATEGORY_MARKERS = {
    "Self-Transcendence": "o",
    "Conservation":       "s",
    "Self-Enhancement":   "^",
    "Openness to Change": "X",
}

for (model_id, model_label), color in zip(MODELS, colors):
    first = True
    for vs in VALUE_ORDER:
        pt = probe_total[(model_id, vs)]
        dt = direct_total[(model_id, vs)]
        if pt == 0 or dt == 0:
            continue
        x = 100 * probe_comply[(model_id, vs)] / pt
        y = 100 * direct_comply[(model_id, vs)] / dt
        marker = CATEGORY_MARKERS[VALUE_TO_CATEGORY[vs]]
        ax.scatter(x, y, color=color, s=90, marker=marker, zorder=3,
                   edgecolors="white", linewidths=0.5,
                   label=model_label if first else "_nolegend_")
        first = False

# Dummy entries for category legend
for cat, marker in CATEGORY_MARKERS.items():
    ax.scatter([], [], marker=marker, color="grey", s=80, label=cat)

# reference line
ax.plot(t, 100 - t, color="grey", linestyle="--", linewidth=1.5,
        label="Consistency line", zorder=1)

ax.set_xlim(0, 105)
ax.set_ylim(0, 105)
ax.set_xlabel("Probe agreement rate (%)", fontsize=21)
ax.set_ylabel("Direct jailbreak ASR (%)", fontsize=21)
ax.tick_params(labelsize=18)
ax.legend(fontsize=16, loc="upper left", framealpha=0.9)
ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
plt.savefig("analysis/value_action_gap.pdf", format="pdf", bbox_inches="tight")
plt.savefig("analysis/value_action_gap.png", dpi=150, bbox_inches="tight")
print("Saved: analysis/value_action_gap.pdf / .png")
plt.show()
