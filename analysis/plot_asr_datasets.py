import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("data/civics_attack_results_v2.json") as f:
    civics = json.load(f)
with open("data/pluriharms_attack_results_v2.json") as f:
    pluriharms = json.load(f)

# Exclude mistral-pixtral (incomplete data)
civics     = [e for e in civics     if e["model"] != "mistral-pixtral"]
pluriharms = [e for e in pluriharms if e["model"] != "mistral-pixtral"]
combined   = civics + pluriharms

MODELS = [
    ("deepseek-r1",     "DeepSeek R1",      "Chinese"),
    ("deepseek-v3.2",   "DeepSeek V3.2",    "Chinese"),
    ("llama3-70b",      "Llama 3.3 70B",    "Western"),
    ("llama4-maverick", "Llama 4 Maverick", "Western"),
    ("mistral-large-3", "Mistral Large 3",  "European"),
    ("qwen3-32b",       "Qwen3 32B",        "Chinese"),
    ("qwen3-235b",      "Qwen3 235B",       "Chinese"),
]

ORIGIN_COLOR = {
    "Chinese":  "#56517E",
    "Western":  "#B46504",
    "European": "#AE4132",
}

CONDITION_LABELS  = ["Direct (d=0)", "Adjacent (d=1)", "Opposing (d=2)"]
CONDITION_HATCHES = ["", "//", "xx"]


def compute_model_stats(data):
    stats = []
    for model_id, label, origin in MODELS:
        entries = [e for e in data if e["model"] == model_id]
        row = [label, origin]
        for dr in [0, 1, 2]:
            valid = [c["comply"] for e in entries for c in e["conditions"]
                     if c["distance_rank"] == dr
                     and c["comply"] != -1 and c["comply"] is not None]
            asr = 100 * sum(1 for v in valid if v == 1) / len(valid) if valid else 0.0
            row.append(asr)
        stats.append(tuple(row))
    return stats  # (label, origin, d0, d1, d2)


def plot_asr(data, title, fname):
    model_stats = compute_model_stats(data)

    bar_width  = 0.22
    group_gap  = 0.15
    x = np.arange(len(model_stats)) * (3 * bar_width + group_gap)

    fig, ax = plt.subplots(figsize=(13, 6))

    for ci, (label, hatch) in enumerate(zip(CONDITION_LABELS, CONDITION_HATCHES)):
        for mi, (m_label, origin, d0, d1, d2) in enumerate(model_stats):
            vals  = [d0, d1, d2]
            color = ORIGIN_COLOR[origin]
            xpos  = x[mi] + ci * bar_width
            ax.bar(xpos, vals[ci], width=bar_width,
                   color=color,
                   alpha=0.85 if ci == 0 else (0.65 if ci == 1 else 0.45),
                   hatch=hatch, edgecolor="white", linewidth=0.6)
            ax.text(xpos, vals[ci] + 0.4, f"{vals[ci]:.1f}",
                    ha="center", va="bottom", fontsize=11, color="#333333")

    group_center = x + bar_width
    ax.set_xticks(group_center)
    ax.set_xticklabels([m[0] for m in model_stats], fontsize=13)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_ylim(0, 40)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cond_patches = [
        mpatches.Patch(facecolor="#888888", alpha=0.85, hatch="",   label="Direct (d=0)"),
        mpatches.Patch(facecolor="#888888", alpha=0.65, hatch="//", label="Adjacent (d=1)"),
        mpatches.Patch(facecolor="#888888", alpha=0.45, hatch="xx", label="Opposing (d=2)"),
    ]
    origin_patches = [
        mpatches.Patch(facecolor=ORIGIN_COLOR["Chinese"],  label="Chinese"),
        mpatches.Patch(facecolor=ORIGIN_COLOR["Western"],  label="Western"),
        mpatches.Patch(facecolor=ORIGIN_COLOR["European"], label="European"),
    ]
    leg1 = ax.legend(handles=cond_patches,   loc="upper right", fontsize=12,
                     title="Condition",    framealpha=0.9)
    leg2 = ax.legend(handles=origin_patches, loc="upper left",  fontsize=12,
                     title="Model Origin", framealpha=0.9)
    ax.add_artist(leg1)

    plt.tight_layout()
    plt.savefig(f"analysis/{fname}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"analysis/{fname}.png", dpi=150,     bbox_inches="tight")
    print(f"Saved: analysis/{fname}.pdf / .png")
    plt.show()


plot_asr(civics,     "ASR by Model and Value Distance — CIVICS",         "asr_civics")
plot_asr(pluriharms, "ASR by Model and Value Distance — PluriHarms",     "asr_pluriharms")
plot_asr(combined,   "ASR by Model and Value Distance — Combined",       "asr_combined")
