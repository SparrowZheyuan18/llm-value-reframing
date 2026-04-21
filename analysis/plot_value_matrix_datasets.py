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

VALUE_ORDER = [
    "Universalism", "Benevolence", "Conformity", "Tradition", "Security",
    "Power", "Achievement", "Self-Direction", "Stimulation", "Hedonism",
]


def build_matrix(entries, distance_rank):
    comply = defaultdict(int)
    total  = defaultdict(int)
    for e in entries:
        vs = e["v_s"]
        for c in e["conditions"]:
            vf = c["v_f"]
            if vf is None or c["comply"] == -1 or c["comply"] is None:
                continue
            if c["distance_rank"] != distance_rank:
                continue
            comply[(vf, vs)] += int(c["comply"] == 1)
            total[(vf, vs)]  += 1

    mat = np.full((len(VALUE_ORDER), len(VALUE_ORDER)), np.nan)
    for i, vf in enumerate(VALUE_ORDER):
        for j, vs in enumerate(VALUE_ORDER):
            n = total[(vf, vs)]
            if n > 0:
                mat[i, j] = 100 * comply[(vf, vs)] / n
    return mat


def plot_matrices(data, dataset_label, fname_prefix):
    for distance_rank, dr_label, dr_suffix in [(1, "Adjacent (d=1)", "d1"),
                                               (2, "Opposing (d=2)", "d2")]:
        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        axes = axes.flatten()

        cmap = plt.cm.YlOrRd
        norm = mcolors.Normalize(vmin=0, vmax=50)

        for ax_idx, (model_id, model_label) in enumerate(MODELS):
            ax  = axes[ax_idx]
            mat = build_matrix([e for e in data if e["model"] == model_id],
                               distance_rank)

            ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

            for i in range(len(VALUE_ORDER)):
                for j in range(len(VALUE_ORDER)):
                    if np.isnan(mat[i, j]):
                        continue
                    val = mat[i, j]
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if val > 30 else "black")

            ax.set_xticks(range(len(VALUE_ORDER)))
            ax.set_xticklabels(VALUE_ORDER, rotation=35, ha="right", fontsize=9)
            ax.set_yticks(range(len(VALUE_ORDER)))
            ax.set_yticklabels(VALUE_ORDER, fontsize=9)
            ax.set_title(model_label, fontsize=12, fontweight="bold", pad=6)

            if ax_idx % 4 == 0:
                ax.set_ylabel("Framing value (v_f)", fontsize=9)
            if ax_idx >= 4:
                ax.set_xlabel("Statement value (v_s)", fontsize=9)

        axes[-1].set_visible(False)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            ax=axes[-1], fraction=0.6, pad=0.05)
        cbar.set_label("ASR (%)", fontsize=11)

        fig.suptitle(
            f"Value Conflict Matrix — {dataset_label} · {dr_label}\n"
            "(rows = framing value v_f, cols = statement value v_s)",
            fontsize=13, fontweight="bold", y=1.01)

        plt.tight_layout()
        fname = f"analysis/value_matrix_{fname_prefix}_{dr_suffix}"
        plt.savefig(f"{fname}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{fname}.png", dpi=150,     bbox_inches="tight")
        print(f"Saved: {fname}.pdf / .png")
        plt.show()


def plot_combined(data):
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.flatten()

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=50)

    for ax_idx, (model_id, model_label) in enumerate(MODELS):
        ax  = axes[ax_idx]
        # d=1 and d=2 combined (distance_rank=None handled by building without filter)
        comply = defaultdict(int)
        total  = defaultdict(int)
        for e in [e for e in data if e["model"] == model_id]:
            vs = e["v_s"]
            for c in e["conditions"]:
                vf = c["v_f"]
                if vf is None or c["comply"] == -1 or c["comply"] is None:
                    continue
                if c["distance_rank"] == 0:
                    continue
                comply[(vf, vs)] += int(c["comply"] == 1)
                total[(vf, vs)]  += 1

        mat = np.full((len(VALUE_ORDER), len(VALUE_ORDER)), np.nan)
        for i, vf in enumerate(VALUE_ORDER):
            for j, vs in enumerate(VALUE_ORDER):
                n = total[(vf, vs)]
                if n > 0:
                    mat[i, j] = 100 * comply[(vf, vs)] / n

        ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        for i in range(len(VALUE_ORDER)):
            for j in range(len(VALUE_ORDER)):
                if np.isnan(mat[i, j]):
                    continue
                val = mat[i, j]
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val > 30 else "black")

        ax.set_xticks(range(len(VALUE_ORDER)))
        ax.set_xticklabels(VALUE_ORDER, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(VALUE_ORDER)))
        ax.set_yticklabels(VALUE_ORDER, fontsize=9)
        ax.set_title(model_label, fontsize=12, fontweight="bold", pad=6)

        if ax_idx % 4 == 0:
            ax.set_ylabel("Framing value (v_f)", fontsize=9)
        if ax_idx >= 4:
            ax.set_xlabel("Statement value (v_s)", fontsize=9)

    axes[-1].set_visible(False)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes[-1], fraction=0.6, pad=0.05)
    cbar.set_label("ASR (%)", fontsize=11)

    fig.suptitle(
        "Value Conflict Matrix — Combined (CIVICS + PluriHarms)\n"
        "(rows = framing value v_f, cols = statement value v_s; d=1 and d=2 combined)",
        fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig("analysis/value_matrix_combined.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("analysis/value_matrix_combined.png", dpi=150,     bbox_inches="tight")
    print("Saved: analysis/value_matrix_combined.pdf / .png")
    plt.show()


plot_combined(combined)
