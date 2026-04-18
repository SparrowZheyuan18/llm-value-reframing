"""
3-Way Comparison using Kaleido XL on all datasets:
1. PluriHarms Original (harmful prompts)
2. PluriHarms Civics (civics-reframed versions)
3. CIVICS Original (separate civics dataset)
"""

import json
import os
from collections import defaultdict
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# All three Kaleido XL results
PLURIHARMS_KALEIDO = os.path.join(PROJECT_DIR, 'data/kaleido_results/pluriharms_original_kaleido.jsonl')
PLURIHARMS_CIVICS_KALEIDO = os.path.join(PROJECT_DIR, 'data/kaleido_results/civics_llm_kaleido.jsonl')
CIVICS_KALEIDO = os.path.join(PROJECT_DIR, 'data/kaleido_results/civics_kaleido_raw.jsonl')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'analysis/dataset_values')

# 39 value categories
VALUE_CATEGORIES = [
    "Right to Privacy and Protection",
    "Freedom of Expression and Speech",
    "Duty to Promote Public Welfare",
    "Duty to Provide Accurate Information",
    "Fairness and Honesty Duties",
    "Right to Information and Accuracy",
    "Autonomy and Bodily Integrity Rights",
    "Health and Well-being",
    "Respect for Others' Beliefs and Preferences",
    "Fairness and Equal Treatment Rights",
    "Honesty and Truthfulness",
    "Right to Security and Financial Security",
    "Cultural Diversity and Inclusion",
    "Right to Education",
    "Equality",
    "Trust and Loyalty",
    "Right to Safety and Self-Defense",
    "Workplace Conduct and Ethics",
    "Right to a Safe and Healthy Environment",
    "Academic and Professional Integrity",
    "Scientific and Technological Advancement",
    "Economic Efficiency and Productivity",
    "Personal and Economic Growth",
    "Transparency and Historical Accuracy",
    "Justice and Fairness",
    "Creativity and Innovation Promotion",
    "Social and Community Cohesion",
    "Property and Housing Rights",
    "Intellectual Property Rights and Duties",
    "Duty to Report Misconduct",
    "Environmental Responsibility",
    "Economic and Financial Stability",
    "Fair Treatment and Cultural Preservation Duties",
    "Merit-Based Achievement System",
    "Preservation and Sanctity of Life",
    "Animal Welfare and Humane Treatment",
    "Work-Life Balance Prioritization",
    "Cultural Tradition and Social Harmony",
    "Selfless Service to Others"
]


def load_kaleido_jsonl(path: str) -> List[Dict]:
    """Load Kaleido results from JSONL file"""
    results = []
    with open(path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def map_to_categories(results: List[Dict], name: str, threshold: float = 0.3) -> Dict:
    """Map VRDs to 39 categories via semantic similarity"""

    # Extract VRDs
    all_vrds = []
    vrd_meta = []

    for result in results:
        for vrd in result.get('vrds', []):
            text = vrd.get('text', '')
            if text:
                all_vrds.append(text)
                vrd_meta.append({
                    'supports': vrd.get('supports', 0),
                    'opposes': vrd.get('opposes', 0)
                })

    print(f"{name}: {len(results)} items, {len(all_vrds)} VRDs")

    if not all_vrds:
        return {'name': name, 'counts': {}, 'valences': {}, 'total': 0, 'num_items': len(results)}

    # Embed
    model = SentenceTransformer('all-mpnet-base-v2')
    print(f"  Embedding {len(all_vrds)} VRDs...")
    cat_embeddings = model.encode(VALUE_CATEGORIES, show_progress_bar=False)
    vrd_embeddings = model.encode(all_vrds, show_progress_bar=True)

    # Compute similarities
    sims = cosine_similarity(vrd_embeddings, cat_embeddings)

    # Assign
    counts = defaultdict(int)
    valences = defaultdict(list)

    for i, (sim_row, meta) in enumerate(zip(sims, vrd_meta)):
        max_idx = np.argmax(sim_row)
        if sim_row[max_idx] >= threshold:
            cat = VALUE_CATEGORIES[max_idx]
            counts[cat] += 1
            valences[cat].append(meta['supports'] - meta['opposes'])

    return {
        'name': name,
        'counts': dict(counts),
        'valences': {k: np.mean(v) for k, v in valences.items()},
        'total': sum(counts.values()),
        'num_items': len(results)
    }


def generate_visualizations(data1: Dict, data2: Dict, data3: Dict):
    """Generate 3-way comparison visualizations"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Calculate percentages
    def get_pcts(data):
        total = data['total']
        if total == 0:
            return {cat: 0 for cat in VALUE_CATEGORIES}
        return {cat: data['counts'].get(cat, 0) / total * 100 for cat in VALUE_CATEGORIES}

    pct1, pct2, pct3 = get_pcts(data1), get_pcts(data2), get_pcts(data3)

    # Sort by total
    sorted_cats = sorted(VALUE_CATEGORIES,
                         key=lambda c: pct1[c] + pct2[c] + pct3[c], reverse=True)

    # 1. Side-by-side percentage bar chart (top 20)
    fig, ax = plt.subplots(figsize=(16, 12))

    top_cats = sorted_cats[:20]
    x = np.arange(len(top_cats))
    width = 0.25

    pcts1 = [pct1[c] for c in top_cats]
    pcts2 = [pct2[c] for c in top_cats]
    pcts3 = [pct3[c] for c in top_cats]

    ax.barh(x - width, pcts1, width, label=data1['name'], color='#2E86AB')
    ax.barh(x, pcts2, width, label=data2['name'], color='#E94F37')
    ax.barh(x + width, pcts3, width, label=data3['name'], color='#44BBA4')

    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel('Value Category', fontsize=12)
    ax.set_title('Value Category Distribution: 3-Way Kaleido XL Comparison (Top 20)', fontsize=14)
    ax.set_yticks(x)
    ax.set_yticklabels([c[:35] + '...' if len(c) > 35 else c for c in top_cats], fontsize=9)
    ax.legend(loc='lower right', fontsize=10)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3way_kaleido_percentages.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Heatmap
    fig, ax = plt.subplots(figsize=(10, 16))

    matrix = np.array([[pct1[c], pct2[c], pct3[c]] for c in sorted_cats])

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([data1['name'], data2['name'], data3['name']], fontsize=10)
    ax.set_yticks(range(len(sorted_cats)))
    ax.set_yticklabels([c[:30] + '...' if len(c) > 30 else c for c in sorted_cats], fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Percentage (%)', fontsize=10)

    ax.set_title('Value Category Heatmap: All 39 Categories (Kaleido XL)', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3way_kaleido_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Radar chart (top 10)
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    top10 = sorted_cats[:10]
    angles = np.linspace(0, 2 * np.pi, len(top10), endpoint=False).tolist()
    angles += angles[:1]

    values1 = [pct1[c] for c in top10] + [pct1[top10[0]]]
    values2 = [pct2[c] for c in top10] + [pct2[top10[0]]]
    values3 = [pct3[c] for c in top10] + [pct3[top10[0]]]

    ax.plot(angles, values1, 'o-', linewidth=2, label=data1['name'], color='#2E86AB')
    ax.fill(angles, values1, alpha=0.15, color='#2E86AB')
    ax.plot(angles, values2, 'o-', linewidth=2, label=data2['name'], color='#E94F37')
    ax.fill(angles, values2, alpha=0.15, color='#E94F37')
    ax.plot(angles, values3, 'o-', linewidth=2, label=data3['name'], color='#44BBA4')
    ax.fill(angles, values3, alpha=0.15, color='#44BBA4')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c[:20] + '...' if len(c) > 20 else c for c in top10], fontsize=9)
    ax.set_title('Top 10 Value Categories: Radar Comparison (Kaleido XL)', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3way_kaleido_radar.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Difference chart: PluriHarms Original vs PluriHarms Civics
    fig, ax = plt.subplots(figsize=(14, 12))

    diffs = [(c, pct2[c] - pct1[c]) for c in sorted_cats]
    diffs_sorted = sorted(diffs, key=lambda x: abs(x[1]), reverse=True)[:20]

    cats = [d[0] for d in diffs_sorted]
    diff_vals = [d[1] for d in diffs_sorted]
    colors = ['#E94F37' if d > 0 else '#2E86AB' for d in diff_vals]

    ax.barh(range(len(cats)), diff_vals, color=colors)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([c[:35] + '...' if len(c) > 35 else c for c in cats], fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Percentage Point Difference (PluriHarms Civics - PluriHarms Original)')
    ax.set_title('Effect of Civics Reframing on Value Distribution', fontsize=14)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E94F37', label='More in Civics Reframing'),
                       Patch(facecolor='#2E86AB', label='More in Original')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3way_kaleido_civics_effect.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualizations to {OUTPUT_DIR}")


def print_summary(data1: Dict, data2: Dict, data3: Dict):
    """Print summary"""

    def get_pcts(data):
        total = data['total']
        if total == 0:
            return {cat: 0 for cat in VALUE_CATEGORIES}
        return {cat: data['counts'].get(cat, 0) / total * 100 for cat in VALUE_CATEGORIES}

    pct1, pct2, pct3 = get_pcts(data1), get_pcts(data2), get_pcts(data3)

    print("\n" + "="*80)
    print("3-WAY KALEIDO XL COMPARISON")
    print("="*80)

    print(f"\nDataset Statistics:")
    print(f"  {data1['name']}: {data1['num_items']} items → {data1['total']} VRDs")
    print(f"  {data2['name']}: {data2['num_items']} items → {data2['total']} VRDs")
    print(f"  {data3['name']}: {data3['num_items']} items → {data3['total']} VRDs")

    for data, pct in [(data1, pct1), (data2, pct2), (data3, pct3)]:
        print(f"\nTop 5 in {data['name']}:")
        for cat in sorted(VALUE_CATEGORIES, key=lambda c: pct[c], reverse=True)[:5]:
            print(f"  {cat}: {pct[cat]:.1f}%")

    # Effect of civics reframing
    print("\n" + "-"*80)
    print("EFFECT OF CIVICS REFRAMING (PluriHarms Civics - PluriHarms Original):")

    diffs = [(c, pct2[c] - pct1[c]) for c in VALUE_CATEGORIES if abs(pct2[c] - pct1[c]) > 1]
    diffs_sorted = sorted(diffs, key=lambda x: x[1], reverse=True)

    print("\n  Increased in civics reframing:")
    for cat, diff in [d for d in diffs_sorted if d[1] > 0][:5]:
        print(f"    +{diff:.1f}pp: {cat}")

    print("\n  Decreased in civics reframing:")
    for cat, diff in [d for d in diffs_sorted if d[1] < 0][:5]:
        print(f"    {diff:.1f}pp: {cat}")

    print("\n" + "="*80)


def main():
    print("Loading Kaleido XL results...")

    # Load all three
    pluriharms = load_kaleido_jsonl(PLURIHARMS_KALEIDO)
    pluriharms_civics = load_kaleido_jsonl(PLURIHARMS_CIVICS_KALEIDO)
    civics = load_kaleido_jsonl(CIVICS_KALEIDO)

    # Map to categories
    print("\nMapping to 39 categories...")
    data1 = map_to_categories(pluriharms, 'PluriHarms Original')
    data2 = map_to_categories(pluriharms_civics, 'PluriHarms Civics')
    data3 = map_to_categories(civics, 'CIVICS Original')

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(data1, data2, data3)

    # Save stats
    stats = {
        'datasets': [
            {'name': d['name'], 'num_items': d['num_items'], 'total_vrds': d['total'], 'counts': d['counts']}
            for d in [data1, data2, data3]
        ]
    }
    with open(os.path.join(OUTPUT_DIR, '3way_kaleido_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print_summary(data1, data2, data3)

    print(f"\nOutput files in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
