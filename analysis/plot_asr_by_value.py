"""
Plot ASR broken down by stated value (v_s) and framing value (v_f).

Outputs:
  analysis/figures/asr_by_vs.png       — ASR by stated value, per model, per distance
  analysis/figures/asr_by_vf.png       — ASR by framing value, per model (d=1 + d=2 combined)
  analysis/figures/asr_vs_freq.png     — scatter: dataset frequency vs ASR (rare-value hypothesis)
  analysis/figures/sanity_circumplex.png — heatmap: v_s x v_f combo frequency and ASR

Usage:
    python analysis/plot_asr_by_value.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
RESULTS_PATH = ROOT / "data" / "civics_attack_results.json"
PROMPTS_PATH = ROOT / "data" / "civics_jailbreak_prompts.json"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["llama3-70b", "deepseek-r1", "qwen3-32b", "mistral-large-3"]
MODEL_LABELS = {
    "llama3-70b":    "Llama 3.3 70B",
    "deepseek-r1":   "DeepSeek R1",
    "qwen3-32b":     "Qwen3 32B",
    "mistral-large-3": "Mistral Large 3",
}
DIST_COLORS = {0: "#888888", 1: "#f4a261", 2: "#e76f51"}
DIST_LABELS = {0: "Direct (d=0)", 1: "Adjacent (d=1)", 2: "Opposing (d=2)"}

ALL_VALUES = [
    "Universalism", "Benevolence", "Conformity", "Tradition",
    "Security", "Power", "Achievement", "Hedonism", "Stimulation", "Self-Direction",
]


def load_data():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        prompts = json.load(f)
    return results, prompts


def compute_asr_by_vs(results):
    """Returns {model: {v_s: {distance_rank: (comply, total)}}}"""
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for r in results:
        model = r["model"]
        v_s = r["v_s"]
        for c in r["conditions"]:
            d = c["distance_rank"]
            if c["comply"] != -1:
                stats[model][v_s][d][1] += 1
                stats[model][v_s][d][0] += c["comply"]
    return stats


def compute_asr_by_vf(results):
    """Returns {model: {v_f: {distance_rank: (comply, total)}}}"""
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for r in results:
        model = r["model"]
        for c in r["conditions"]:
            if c["v_f"] is None:
                continue
            v_f = c["v_f"]
            d = c["distance_rank"]
            if c["comply"] != -1:
                stats[model][v_f][d][1] += 1
                stats[model][v_f][d][0] += c["comply"]
    return stats


def compute_vs_freq(prompts):
    """Count how many entries each v_s appears in."""
    freq = defaultdict(int)
    for p in prompts:
        freq[p["v_s"]] += 1
    return freq


def compute_vf_freq(prompts):
    """Count how many conditions each v_f appears in."""
    freq = defaultdict(int)
    for p in prompts:
        for c in p["conditions"]:
            if c["v_f"]:
                freq[c["v_f"]] += 1
    return freq


# ---------------------------------------------------------------------------
# Plot 1: ASR by v_s, per model, grouped bar (one subplot per model)
# ---------------------------------------------------------------------------
def plot_asr_by_vs(stats, vs_freq):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    fig.suptitle("ASR by Stated Value (v_s) × Distance", fontsize=15, fontweight="bold", y=1.01)

    # Gather all v_s that appear in results
    all_vs_in_data = sorted(
        {v_s for m in stats.values() for v_s in m.keys()},
        key=lambda v: -vs_freq.get(v, 0)
    )

    for ax, model in zip(axes.flat, MODELS):
        model_stats = stats.get(model, {})
        x = np.arange(len(all_vs_in_data))
        width = 0.25

        for i, d in enumerate([0, 1, 2]):
            asrs = []
            for v_s in all_vs_in_data:
                comply, total = model_stats.get(v_s, {}).get(d, [0, 0])
                asrs.append(comply / total if total > 0 else 0)
            bars = ax.bar(x + i * width, asrs, width, label=DIST_LABELS[d],
                          color=DIST_COLORS[d], alpha=0.85)

        # Annotate with dataset frequency
        for xi, v_s in enumerate(all_vs_in_data):
            ax.text(xi + width, -0.045, f"n={vs_freq.get(v_s, 0)}",
                    ha="center", fontsize=7, color="#555555", rotation=0)

        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(all_vs_in_data, rotation=35, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_ylim(-0.06, 0.65)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "asr_by_vs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: ASR by v_f, per model, for d=1 and d=2 (skip d=0, no v_f)
# ---------------------------------------------------------------------------
def plot_asr_by_vf(stats, vf_freq):
    # Which v_f values appear?
    all_vf = sorted(
        {v_f for m in stats.values() for v_f in m.keys()},
        key=lambda v: -vf_freq.get(v, 0)
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    fig.suptitle("ASR by Framing Value (v_f) × Distance", fontsize=15, fontweight="bold", y=1.01)

    for ax, model in zip(axes.flat, MODELS):
        model_stats = stats.get(model, {})
        x = np.arange(len(all_vf))
        width = 0.35

        for i, d in enumerate([1, 2]):
            asrs = []
            for v_f in all_vf:
                comply, total = model_stats.get(v_f, {}).get(d, [0, 0])
                asrs.append(comply / total if total > 0 else 0)
            ax.bar(x + i * width, asrs, width, label=DIST_LABELS[d],
                   color=DIST_COLORS[d], alpha=0.85)

        for xi, v_f in enumerate(all_vf):
            ax.text(xi + width / 2, -0.045, f"n={vf_freq.get(v_f, 0)}",
                    ha="center", fontsize=7, color="#555555")

        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(all_vf, rotation=35, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_ylim(-0.06, 0.75)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "asr_by_vf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Scatter — dataset frequency vs. mean ASR (rare-value hypothesis)
# ---------------------------------------------------------------------------
def plot_freq_vs_asr(vs_stats, vs_freq, vf_stats, vf_freq):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Dataset Frequency vs. ASR: Is Rare = More Susceptible?",
                 fontsize=13, fontweight="bold")

    # Left: v_s (stated value)
    ax = axes[0]
    ax.set_title("By Stated Value (v_s)", fontsize=11)
    for v_s in ALL_VALUES:
        freq = vs_freq.get(v_s, 0)
        if freq == 0:
            continue
        # Average ASR at d=2 across all models
        asrs = []
        for model in MODELS:
            comply, total = vs_stats.get(model, {}).get(v_s, {}).get(2, [0, 0])
            if total > 0:
                asrs.append(comply / total)
        if not asrs:
            continue
        mean_asr = np.mean(asrs)
        ax.scatter(freq, mean_asr, s=80, zorder=5)
        ax.annotate(v_s, (freq, mean_asr), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)

    ax.set_xlabel("# entries in dataset (v_s frequency)", fontsize=10)
    ax.set_ylabel("Mean ASR at d=2 (opposing), avg over 4 models", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.grid(alpha=0.3)

    # Right: v_f (framing value)
    ax = axes[1]
    ax.set_title("By Framing Value (v_f)", fontsize=11)
    for v_f in ALL_VALUES:
        freq = vf_freq.get(v_f, 0)
        if freq == 0:
            continue
        asrs_d1, asrs_d2 = [], []
        for model in MODELS:
            c1, t1 = vf_stats.get(model, {}).get(v_f, {}).get(1, [0, 0])
            c2, t2 = vf_stats.get(model, {}).get(v_f, {}).get(2, [0, 0])
            if t1 > 0:
                asrs_d1.append(c1 / t1)
            if t2 > 0:
                asrs_d2.append(c2 / t2)
        # Combined mean
        combined = asrs_d1 + asrs_d2
        if not combined:
            continue
        mean_asr = np.mean(combined)
        ax.scatter(freq, mean_asr, s=80, zorder=5)
        ax.annotate(v_f, (freq, mean_asr), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)

    ax.set_xlabel("# conditions in dataset (v_f frequency)", fontsize=10)
    ax.set_ylabel("Mean ASR (d=1+d=2 combined), avg over 4 models", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "asr_vs_freq.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Print: v_s breakdown table
# ---------------------------------------------------------------------------
def print_vs_table(stats, vs_freq):
    print("\n" + "=" * 80)
    print("ASR by Stated Value (v_s) — averaged across 4 models")
    print("=" * 80)
    print(f"{'Value':<18} {'n':>5}  {'d=0':>7}  {'d=1':>7}  {'d=2':>7}")
    print("-" * 55)

    all_vs = sorted(vs_freq.keys(), key=lambda v: -vs_freq[v])
    for v_s in all_vs:
        n = vs_freq[v_s]
        row = [v_s, n]
        for d in [0, 1, 2]:
            asrs = []
            for model in MODELS:
                comply, total = stats.get(model, {}).get(v_s, {}).get(d, [0, 0])
                if total > 0:
                    asrs.append(comply / total)
            mean = np.mean(asrs) if asrs else float("nan")
            row.append(mean)
        print(f"{row[0]:<18} {row[1]:>5}  {row[2]:>6.1%}  {row[3]:>6.1%}  {row[4]:>6.1%}")


def print_vf_table(stats, vf_freq):
    print("\n" + "=" * 80)
    print("ASR by Framing Value (v_f) — averaged across 4 models")
    print("=" * 80)
    print(f"{'Value':<18} {'n':>6}  {'d=1 (adj)':>10}  {'d=2 (opp)':>10}")
    print("-" * 55)

    all_vf = sorted(vf_freq.keys(), key=lambda v: -vf_freq[v])
    for v_f in all_vf:
        n = vf_freq[v_f]
        asrs_d1, asrs_d2 = [], []
        for model in MODELS:
            c1, t1 = stats.get(model, {}).get(v_f, {}).get(1, [0, 0])
            c2, t2 = stats.get(model, {}).get(v_f, {}).get(2, [0, 0])
            if t1 > 0:
                asrs_d1.append(c1 / t1)
            if t2 > 0:
                asrs_d2.append(c2 / t2)
        m1 = np.mean(asrs_d1) if asrs_d1 else float("nan")
        m2 = np.mean(asrs_d2) if asrs_d2 else float("nan")
        print(f"{v_f:<18} {n:>6}  {m1:>9.1%}  {m2:>9.1%}")


# ---------------------------------------------------------------------------
def main():
    results, prompts = load_data()

    vs_stats = compute_asr_by_vs(results)
    vf_stats = compute_asr_by_vf(results)
    vs_freq = compute_vs_freq(prompts)
    vf_freq = compute_vf_freq(prompts)

    print_vs_table(vs_stats, vs_freq)
    print_vf_table(vf_stats, vf_freq)

    plot_asr_by_vs(vs_stats, vs_freq)
    plot_asr_by_vf(vf_stats, vf_freq)
    plot_freq_vs_asr(vs_stats, vs_freq, vf_stats, vf_freq)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
