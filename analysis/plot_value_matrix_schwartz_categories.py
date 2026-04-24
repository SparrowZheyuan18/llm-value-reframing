import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

with open("data/civics_attack_results_v2.json") as f:
    civics = json.load(f)
with open("data/pluriharms_attack_results_v2.json") as f:
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

# Schwartz higher-order categories (in circumplex order)
CATEGORIES = [
    "Self-Transcendence",
    "Conservation",
    "Self-Enhancement",
    "Openness to Change",
]

<<<<<<< HEAD
CATEGORY_ABBREV = {
    "Self-Transcendence": "ST",
    "Conservation":       "CO",
    "Self-Enhancement":   "SE",
    "Openness to Change": "OC",
}

=======
>>>>>>> 848011b98b529f573aed7cb41b8b7546e8af6011
VALUE_TO_CATEGORY = {
    "Universalism":    "Self-Transcendence",
    "Benevolence":     "Self-Transcendence",
    "Conformity":      "Conservation",
    "Tradition":       "Conservation",
    "Security":        "Conservation",
    "Power":           "Self-Enhancement",
    "Achievement":     "Self-Enhancement",
    "Self-Direction":  "Openness to Change",
    "Stimulation":     "Openness to Change",
    "Hedonism":        "Openness to Change",
}


def build_category_matrix(entries, distance_rank=None):
    """
    Aggregate ASR at the higher-order category level.
    matrix[i][j] = ASR when CATEGORIES[i] frames attack on CATEGORIES[j] content
    distance_rank: None = all framed (excluding direct), 1 = adjacent, 2 = opposing
    """
    comply = defaultdict(int)
    total  = defaultdict(int)

    for e in entries:
        vs_cat = VALUE_TO_CATEGORY.get(e["v_s"])
        if vs_cat is None:
            continue
        for c in e["conditions"]:
            vf = c["v_f"]
            if vf is None or c["comply"] == -1 or c["comply"] is None:
                continue
            if c["distance_rank"] == 0:
                continue
            if distance_rank is not None and c["distance_rank"] != distance_rank:
                continue
            vf_cat = VALUE_TO_CATEGORY.get(vf)
            if vf_cat is None:
                continue
            comply[(vf_cat, vs_cat)] += int(c["comply"] == 1)
            total[(vf_cat, vs_cat)]  += 1

    n = len(CATEGORIES)
    mat = np.full((n, n), np.nan)
    for i, vf_cat in enumerate(CATEGORIES):
        for j, vs_cat in enumerate(CATEGORIES):
            t = total[(vf_cat, vs_cat)]
            if t > 0:
                mat[i, j] = 100 * comply[(vf_cat, vs_cat)] / t
    return mat


def plot_category_matrices(data, dataset_label, fname_prefix, distance_rank=None, dr_label="All framed", dr_suffix="all"):
<<<<<<< HEAD
    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
=======
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
>>>>>>> 848011b98b529f573aed7cb41b8b7546e8af6011
    axes = axes.flatten()

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=50)

    for ax_idx, (model_id, model_label) in enumerate(MODELS):
        ax  = axes[ax_idx]
        mat = build_category_matrix(
            [e for e in data if e["model"] == model_id],
            distance_rank=distance_rank,
        )

<<<<<<< HEAD
        ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")
=======
        ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
>>>>>>> 848011b98b529f573aed7cb41b8b7546e8af6011

        for i in range(len(CATEGORIES)):
            for j in range(len(CATEGORIES)):
                if np.isnan(mat[i, j]):
                    continue
                val = mat[i, j]
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
<<<<<<< HEAD
                        fontsize=14, fontweight="bold",
                        color="white" if val > 30 else "black")

        abbrevs = [CATEGORY_ABBREV[c] for c in CATEGORIES]
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels(abbrevs, rotation=0, ha="center", fontsize=13)
        ax.set_yticks(range(len(CATEGORIES)))
        ax.set_yticklabels(abbrevs, fontsize=13)
        ax.set_title(model_label, fontsize=18, fontweight="bold", pad=6)

        if ax_idx % 4 == 0:
            ax.set_ylabel("Framing category (v_f)", fontsize=13)
        if ax_idx >= 4:
            ax.set_xlabel("Statement category (v_s)", fontsize=13)
=======
                        fontsize=10, fontweight="bold",
                        color="white" if val > 30 else "black")

        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels(CATEGORIES, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(CATEGORIES)))
        ax.set_yticklabels(CATEGORIES, fontsize=9)
        ax.set_title(model_label, fontsize=12, fontweight="bold", pad=6)

        if ax_idx % 4 == 0:
            ax.set_ylabel("Framing category (v_f)", fontsize=9)
        if ax_idx >= 4:
            ax.set_xlabel("Statement category (v_s)", fontsize=9)
>>>>>>> 848011b98b529f573aed7cb41b8b7546e8af6011

    axes[-1].set_visible(False)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes[-1], fraction=0.6, pad=0.05)
<<<<<<< HEAD
    cbar.set_label("ASR (%)", fontsize=16)
    cbar.ax.tick_params(labelsize=13)
=======
    cbar.set_label("ASR (%)", fontsize=11)

    fig.suptitle(
        f"Value Conflict Matrix — {dataset_label} · {dr_label}\n"
        "(rows = framing category, cols = statement category; Schwartz higher-order)",
        fontsize=13, fontweight="bold", y=1.01)
>>>>>>> 848011b98b529f573aed7cb41b8b7546e8af6011

    plt.tight_layout()
    fname = f"analysis/value_matrix_schwartz_{fname_prefix}_{dr_suffix}"
    plt.savefig(f"{fname}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{fname}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}.pdf / .png")
    plt.show()


# Combined dataset, all framed (d=1 + d=2)
plot_category_matrices(combined, "Combined (CIVICS + PluriHarms)", "combined",
                       distance_rank=None, dr_label="All framed (d=1+2)", dr_suffix="all")

# Adjacent only
plot_category_matrices(combined, "Combined (CIVICS + PluriHarms)", "combined",
                       distance_rank=1, dr_label="Adjacent (d=1)", dr_suffix="d1")

# Opposing only
plot_category_matrices(combined, "Combined (CIVICS + PluriHarms)", "combined",
                       distance_rank=2, dr_label="Opposing (d=2)", dr_suffix="d2")
