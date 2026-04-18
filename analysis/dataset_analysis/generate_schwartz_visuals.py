#!/usr/bin/env python3
"""
Generate visualizations comparing Schwartz values distribution
between PluriHarms-Civics and Civics datasets.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math

# Load datasets
with open('data/pluriharms_civics_schwartz.json', 'r') as f:
    pluriharms_data = json.load(f)

with open('data/civics_dataset.json', 'r') as f:
    civics_data = json.load(f)

# Define all Schwartz values (10 basic values)
SCHWARTZ_VALUES = [
    'Self-Direction', 'Stimulation', 'Hedonism', 'Achievement', 'Power',
    'Security', 'Conformity', 'Tradition', 'Benevolence', 'Universalism'
]

# Define higher-order values
SCHWARTZ_HIGHER_ORDER = [
    'Openness to Change', 'Self-Enhancement', 'Conservation', 'Self-Transcendence'
]

def extract_pluriharms_values(data):
    """Extract Schwartz values from PluriHarms dataset."""
    basic_values = []
    higher_order = []

    for entry in data:
        for statement in entry.get('generated_statements', []):
            basic_values.extend(statement.get('schwartz_values', []))
            higher_order.extend(statement.get('schwartz_higher_order', []))

    return basic_values, higher_order

def extract_civics_values(data):
    """Extract Schwartz values from Civics dataset."""
    basic_values = []
    higher_order = []

    for entry in data:
        basic_values.extend(entry.get('Schwartz Values', []))
        higher_order.extend(entry.get('Schwartz Higher-Order', []))

    return basic_values, higher_order

# Extract values
ph_basic, ph_higher = extract_pluriharms_values(pluriharms_data)
civics_basic, civics_higher = extract_civics_values(civics_data)

# Count values
ph_basic_counts = Counter(ph_basic)
ph_higher_counts = Counter(ph_higher)
civics_basic_counts = Counter(civics_basic)
civics_higher_counts = Counter(civics_higher)

# Normalize to percentages
def normalize_counts(counts, all_values):
    total = sum(counts.values())
    if total == 0:
        return [0] * len(all_values)
    return [counts.get(v, 0) / total * 100 for v in all_values]

ph_basic_pct = normalize_counts(ph_basic_counts, SCHWARTZ_VALUES)
civics_basic_pct = normalize_counts(civics_basic_counts, SCHWARTZ_VALUES)
ph_higher_pct = normalize_counts(ph_higher_counts, SCHWARTZ_HIGHER_ORDER)
civics_higher_pct = normalize_counts(civics_higher_counts, SCHWARTZ_HIGHER_ORDER)

# Print stats
print("=" * 60)
print("SCHWARTZ VALUES DISTRIBUTION COMPARISON")
print("=" * 60)
print(f"\nPluriHarms-Civics: {len(ph_basic)} basic value assignments")
print(f"Civics: {len(civics_basic)} basic value assignments")
print(f"\nPluriHarms-Civics: {len(ph_higher)} higher-order assignments")
print(f"Civics: {len(civics_higher)} higher-order assignments")

# ============================================================
# FIGURE 1: Radar Chart for Basic Schwartz Values
# ============================================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Number of variables
N = len(SCHWARTZ_VALUES)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]  # Complete the loop

# Data
ph_vals = ph_basic_pct + [ph_basic_pct[0]]
civics_vals = civics_basic_pct + [civics_basic_pct[0]]

# Plot
ax.plot(angles, ph_vals, 'o-', linewidth=2, label='PluriHarms-Civics', color='#e74c3c')
ax.fill(angles, ph_vals, alpha=0.25, color='#e74c3c')
ax.plot(angles, civics_vals, 'o-', linewidth=2, label='Civics', color='#3498db')
ax.fill(angles, civics_vals, alpha=0.25, color='#3498db')

# Labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(SCHWARTZ_VALUES, size=11)
ax.set_title('Schwartz Basic Values Distribution:\nPluriHarms-Civics vs Civics',
             size=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig('visual_schwartz_radar_basic.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: visual_schwartz_radar_basic.png")

# ============================================================
# FIGURE 2: Radar Chart for Higher-Order Values
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

N = len(SCHWARTZ_HIGHER_ORDER)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

ph_vals = ph_higher_pct + [ph_higher_pct[0]]
civics_vals = civics_higher_pct + [civics_higher_pct[0]]

ax.plot(angles, ph_vals, 'o-', linewidth=2, label='PluriHarms-Civics', color='#e74c3c')
ax.fill(angles, ph_vals, alpha=0.25, color='#e74c3c')
ax.plot(angles, civics_vals, 'o-', linewidth=2, label='Civics', color='#3498db')
ax.fill(angles, civics_vals, alpha=0.25, color='#3498db')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(SCHWARTZ_HIGHER_ORDER, size=11)
ax.set_title('Schwartz Higher-Order Values Distribution:\nPluriHarms-Civics vs Civics',
             size=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig('visual_schwartz_radar_higher_order.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visual_schwartz_radar_higher_order.png")

# ============================================================
# FIGURE 3: Side-by-side Bar Chart for Basic Values
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(SCHWARTZ_VALUES))
width = 0.35

bars1 = ax.bar(x - width/2, ph_basic_pct, width, label='PluriHarms-Civics', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, civics_basic_pct, width, label='Civics', color='#3498db', alpha=0.8)

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xlabel('Schwartz Basic Values', fontsize=12)
ax.set_title('Schwartz Basic Values: PluriHarms-Civics vs Civics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(SCHWARTZ_VALUES, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('visual_schwartz_bar_basic.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visual_schwartz_bar_basic.png")

# ============================================================
# FIGURE 4: Side-by-side Bar Chart for Higher-Order Values
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(SCHWARTZ_HIGHER_ORDER))
width = 0.35

bars1 = ax.bar(x - width/2, ph_higher_pct, width, label='PluriHarms-Civics', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, civics_higher_pct, width, label='Civics', color='#3498db', alpha=0.8)

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xlabel('Schwartz Higher-Order Values', fontsize=12)
ax.set_title('Schwartz Higher-Order Values: PluriHarms-Civics vs Civics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(SCHWARTZ_HIGHER_ORDER, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('visual_schwartz_bar_higher_order.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visual_schwartz_bar_higher_order.png")

# ============================================================
# FIGURE 5: Heatmap Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

data_matrix = np.array([ph_basic_pct, civics_basic_pct])

im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

ax.set_xticks(np.arange(len(SCHWARTZ_VALUES)))
ax.set_yticks(np.arange(2))
ax.set_xticklabels(SCHWARTZ_VALUES, rotation=45, ha='right')
ax.set_yticklabels(['PluriHarms-Civics', 'Civics'])

# Add text annotations
for i in range(2):
    for j in range(len(SCHWARTZ_VALUES)):
        text = ax.text(j, i, f'{data_matrix[i, j]:.1f}%',
                       ha='center', va='center', color='black', fontsize=9)

ax.set_title('Schwartz Basic Values Distribution Heatmap', fontsize=14, fontweight='bold')
fig.colorbar(im, ax=ax, label='Percentage (%)')

plt.tight_layout()
plt.savefig('visual_schwartz_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visual_schwartz_heatmap.png")

# ============================================================
# FIGURE 6: Difference Chart (PluriHarms - Civics)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

differences = [ph - cv for ph, cv in zip(ph_basic_pct, civics_basic_pct)]
colors = ['#e74c3c' if d > 0 else '#3498db' for d in differences]

bars = ax.bar(SCHWARTZ_VALUES, differences, color=colors, alpha=0.8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

ax.set_ylabel('Difference in Percentage Points', fontsize=12)
ax.set_xlabel('Schwartz Basic Values', fontsize=12)
ax.set_title('Difference in Schwartz Value Distribution\n(PluriHarms-Civics minus Civics)',
             fontsize=14, fontweight='bold')
ax.set_xticklabels(SCHWARTZ_VALUES, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', alpha=0.8, label='More in PluriHarms-Civics'),
                   Patch(facecolor='#3498db', alpha=0.8, label='More in Civics')]
ax.legend(handles=legend_elements, loc='upper right')

# Add value labels
for bar, diff in zip(bars, differences):
    height = bar.get_height()
    ax.annotate(f'{diff:+.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig('visual_schwartz_difference.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visual_schwartz_difference.png")

# ============================================================
# FIGURE 7: Pie Charts Side by Side
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Colors for values
colors = plt.cm.Set3(np.linspace(0, 1, len(SCHWARTZ_VALUES)))

# PluriHarms pie
axes[0].pie(ph_basic_pct, labels=SCHWARTZ_VALUES, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[0].set_title('PluriHarms-Civics\nSchwartz Values Distribution', fontsize=12, fontweight='bold')

# Civics pie
axes[1].pie(civics_basic_pct, labels=SCHWARTZ_VALUES, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[1].set_title('Civics\nSchwartz Values Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visual_schwartz_pie_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visual_schwartz_pie_comparison.png")

# ============================================================
# Print Summary Statistics
# ============================================================
print("\n" + "=" * 60)
print("BASIC VALUE PERCENTAGES")
print("=" * 60)
print(f"{'Value':<20} {'PluriHarms-Civics':>18} {'Civics':>12} {'Difference':>12}")
print("-" * 60)
for i, val in enumerate(SCHWARTZ_VALUES):
    diff = ph_basic_pct[i] - civics_basic_pct[i]
    print(f"{val:<20} {ph_basic_pct[i]:>17.1f}% {civics_basic_pct[i]:>11.1f}% {diff:>+11.1f}")

print("\n" + "=" * 60)
print("HIGHER-ORDER VALUE PERCENTAGES")
print("=" * 60)
print(f"{'Value':<25} {'PluriHarms-Civics':>18} {'Civics':>12} {'Difference':>12}")
print("-" * 60)
for i, val in enumerate(SCHWARTZ_HIGHER_ORDER):
    diff = ph_higher_pct[i] - civics_higher_pct[i]
    print(f"{val:<25} {ph_higher_pct[i]:>17.1f}% {civics_higher_pct[i]:>11.1f}% {diff:>+11.1f}")

print("\n" + "=" * 60)
print("TOP 3 VALUES BY DATASET")
print("=" * 60)
ph_sorted = sorted(zip(SCHWARTZ_VALUES, ph_basic_pct), key=lambda x: x[1], reverse=True)
civics_sorted = sorted(zip(SCHWARTZ_VALUES, civics_basic_pct), key=lambda x: x[1], reverse=True)

print("\nPluriHarms-Civics Top 3:")
for i, (val, pct) in enumerate(ph_sorted[:3], 1):
    print(f"  {i}. {val}: {pct:.1f}%")

print("\nCivics Top 3:")
for i, (val, pct) in enumerate(civics_sorted[:3], 1):
    print(f"  {i}. {val}: {pct:.1f}%")

print("\n" + "=" * 60)
print("All visualizations saved successfully!")
print("=" * 60)
