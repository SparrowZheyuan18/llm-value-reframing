import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

with open("data/attack_results.json") as f:
    data = json.load(f)

MODELS = [
    ("deepseek-r1",     "DeepSeek R1"),
    ("deepseek-v3.2",   "DeepSeek V3.2"),
    ("llama3-70b",      "Llama 3.3 70B"),
    ("llama4-maverick", "Llama 4 Maverick"),
    ("mistral-large-3", "Mistral Large 3"),
    ("qwen3-32b",       "Qwen3 32B"),
    ("qwen3-235b",      "Qwen3 235B"),
]

# Ordered by Schwartz circumplex (roughly): Self-Transcendence → Conservation → Self-Enhancement → Openness
VALUE_ORDER = [
    "Universalism",
    "Benevolence",
    "Conformity",
    "Tradition",
    "Security",
    "Power",
    "Achievement",
    "Self-Direction",
    "Stimulation",
    "Hedonism",
]

VS_VALUES = VALUE_ORDER  # all 10, empty columns where no data
VF_VALUES = VALUE_ORDER  # all 10, empty rows where no data


def build_matrix(entries, vs_vals, vf_vals, distance_rank=None):
    """
    Returns matrix where:
      matrix[i][j] = ASR when vf_vals[i] frames attack on vs_vals[j] content
    distance_rank: None = all framed, 1 = adjacent only, 2 = opposing only
    """
    comply = defaultdict(int)
    total  = defaultdict(int)

    for e in entries:
        vs = e["v_s"]
        for c in e["conditions"]:
            vf = c["v_f"]
            if vf is None or c["comply"] == -1 or c["comply"] is None:
                continue
            if distance_rank is not None and c["distance_rank"] != distance_rank:
                continue
            comply[(vf, vs)] += int(c["comply"] == 1)
            total[(vf, vs)]  += 1

    n_vf, n_vs = len(vf_vals), len(vs_vals)
    mat = np.full((n_vf, n_vs), np.nan)

    for i, vf in enumerate(vf_vals):
        for j, vs in enumerate(vs_vals):
            n = total[(vf, vs)]
            if n > 0:
                mat[i, j] = 100 * comply[(vf, vs)] / n

    return mat


def plot_matrices(distance_rank, title_suffix, fname_suffix):
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.flatten()

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=50)

    for ax_idx, (model_id, model_label) in enumerate(MODELS):
        ax = axes[ax_idx]
        entries = [e for e in data if e["model"] == model_id]
        mat = build_matrix(entries, VS_VALUES, VF_VALUES, distance_rank=distance_rank)

        ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        for i in range(len(VF_VALUES)):
            for j in range(len(VS_VALUES)):
                if np.isnan(mat[i, j]):
                    continue
                val = mat[i, j]
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="white" if val > 30 else "black",
                        fontweight="bold")

        ax.set_xticks(range(len(VS_VALUES)))
        ax.set_xticklabels(VS_VALUES, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(VF_VALUES)))
        ax.set_yticklabels(VF_VALUES, fontsize=9)
        ax.set_title(model_label, fontsize=12, fontweight="bold", pad=6)

        if ax_idx % 4 == 0:
            ax.set_ylabel("Framing value (v_f)", fontsize=9)
        if ax_idx >= 4:
            ax.set_xlabel("Statement value (v_s)", fontsize=9)

    axes[-1].set_visible(False)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes[-1], fraction=0.6, pad=0.05)
    cbar.set_label("ASR (%)", fontsize=11)

    fig.suptitle(f"Value Conflict Matrix — {title_suffix}\n"
                 "(rows = framing value v_f, cols = statement value v_s)",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(f"analysis/value_matrix_{fname_suffix}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"analysis/value_matrix_{fname_suffix}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: analysis/value_matrix_{fname_suffix}.pdf / .png")
    plt.show()


plot_matrices(distance_rank=1, title_suffix="Adjacent values (d=1)",  fname_suffix="d1")
plot_matrices(distance_rank=2, title_suffix="Opposing values (d=2)",  fname_suffix="d2")
