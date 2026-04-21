"""
Filter pluriharms_civics_schwartz.json to single-HO entries, keeping all basic
values under the chosen HO — mirrors the CIVICS single-HO filtering approach.

For each statement (all stances):
  1. Count how many labeled basic values fall under each higher-order category
  2. Keep the HO with the most basic values (tie-break: earliest in the list)
  3. Retain all basic values under the chosen HO as v_s candidates

This produces one row per (statement, v_s) pair, ready for generate_jailbreaks.

Input:  data/pluriharms_civics_schwartz.json
Output: data/pluriharms_filtered.json

Usage:
    python statements_gen/filter_pluriharms.py
"""

import json
from collections import defaultdict
from pathlib import Path

INPUT_PATH  = Path(__file__).parent.parent / "data" / "pluriharms_civics_schwartz.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "pluriharms_filtered.json"

VALUE_TO_HO: dict[str, str] = {
    "Universalism":   "Self-Transcendence",
    "Benevolence":    "Self-Transcendence",
    "Security":       "Conservation",
    "Conformity":     "Conservation",
    "Tradition":      "Conservation",
    "Power":          "Self-Enhancement",
    "Achievement":    "Self-Enhancement",
    "Hedonism":       "Self-Enhancement",
    "Self-Direction": "Openness to Change",
    "Stimulation":    "Openness to Change",
}


def pick_best_ho(svs: list[str], hos: list[str]) -> tuple[str, list[str]] | None:
    """Return (chosen_ho, filtered_basic_values) or None if no valid mapping."""
    ho_to_bvs: dict[str, list[str]] = defaultdict(list)
    for v in svs:
        ho = VALUE_TO_HO.get(v)
        if ho:
            ho_to_bvs[ho].append(v)

    if not ho_to_bvs:
        return None

    def priority(ho: str) -> tuple[int, int]:
        count = len(ho_to_bvs[ho])
        order = hos.index(ho) if ho in hos else 99
        return (-count, order)  # most BVs first, then earliest in original list

    best_ho = min(ho_to_bvs.keys(), key=priority)
    return best_ho, ho_to_bvs[best_ho]


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    results = []
    skipped = 0

    for q in data:
        for s in q["generated_statements"]:
            svs = s.get("schwartz_values", [])
            hos = s.get("schwartz_higher_order", [])

            if not svs:
                skipped += 1
                continue

            out = pick_best_ho(svs, hos)
            if out is None:
                skipped += 1
                continue

            best_ho, kept_bvs = out
            results.append({
                "Question_Index": q["Question_Index"],
                "topic": q["topic"],
                "Original_Prompt": q["Original_Prompt"],
                "statement": s["statement"],
                "stance_polarity": s["stance_polarity"],
                "schwartz_higher_order": best_ho,
                "schwartz_values": kept_bvs,
                # keep originals for reference
                "schwartz_values_all": svs,
                "schwartz_higher_order_all": hos,
            })

    results.sort(key=lambda r: (r["Question_Index"], r["stance_polarity"]))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    from collections import Counter
    ho_dist = Counter(r["schwartz_higher_order"] for r in results)
    bv_dist = Counter(v for r in results for v in r["schwartz_values"])
    n_bvs = Counter(len(r["schwartz_values"]) for r in results)
    total_pairs = sum(len(r["schwartz_values"]) for r in results)

    print(f"Input questions: {len(data)}")
    print(f"Output statements (all stances, single-HO filtered): {len(results)}")
    print(f"Skipped (no valid Schwartz mapping): {skipped}")
    print(f"Total (statement, v_s) work items: {total_pairs}")
    print(f"\nHO distribution:")
    for ho, c in ho_dist.most_common():
        print(f"  {ho}: {c}")
    print(f"\nBasic values per statement:")
    for n, c in sorted(n_bvs.items()):
        print(f"  {n} value(s): {c} statements")
    print(f"\nBasic value work item distribution:")
    for v, c in bv_dist.most_common():
        print(f"  {v}: {c}")
    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
