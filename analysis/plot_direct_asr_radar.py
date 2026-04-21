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

# Compute probe agreement rate per model per v_s
def compute_probe_asr(data):
    comply = defaultdict(int)
    total  = defaultdict(int)
    for e in data:
        vs    = e["v_s"]
        probe = e.get("probe", {})
        c     = probe.get("comply")
        if c == -1 or c is None:
            continue
        comply[(e["model"], vs)] += int(c == 1)
        total[(e["model"], vs)]  += 1
    return comply, total

comply, total = compute_probe_asr(combined)

# Build agreement rate matrix: models x values
asr = {}
for model_id, _ in MODELS:
    asr[model_id] = []
    for vs in VALUE_ORDER:
        n = total[(model_id, vs)]
        asr[model_id].append(100 * comply[(model_id, vs)] / n if n > 0 else np.nan)

# Radar chart — all models on one plot
N = len(VALUE_ORDER)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

colors = plt.cm.tab10(np.linspace(0, 0.9, len(MODELS)))

for (model_id, model_label), color in zip(MODELS, colors):
    values = asr[model_id] + asr[model_id][:1]
    ax.plot(angles, values, color=color, linewidth=2, label=model_label)
    ax.fill(angles, values, color=color, alpha=0.07)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(VALUE_ORDER, fontsize=10)
ax.set_ylim(0, 100)
ax.set_yticks([25, 50, 75, 100])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="grey")
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

fig.suptitle(
    "Probe Agreement Rate by Statement Value (v_s)\n"
    "Combined dataset (CIVICS + PluriHarms)",
    fontsize=13, fontweight="bold", y=1.04,
)
plt.tight_layout()
plt.savefig("analysis/direct_asr_radar.pdf", format="pdf", bbox_inches="tight")
plt.savefig("analysis/direct_asr_radar.png", dpi=150, bbox_inches="tight")
print("Saved: analysis/direct_asr_radar.pdf / .png")
plt.show()
