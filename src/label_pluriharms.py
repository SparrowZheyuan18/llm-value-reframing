"""
Label each PluriHarms-Civics statement with a SINGLE primary Schwartz basic value.

Unlike label_statements.py (which returns all applicable values), this script asks
the LLM to pick the ONE most central basic value per statement. This gives a clean
single v_s for each (statement, stance) pair needed by generate_jailbreaks.

Input:  data/pluriharms_civics_schwartz.json  (already has multi-value labels)
Output: data/pluriharms_schwartz_primary.json  (new file, adds "schwartz_primary" field)

Only processes restrictive + supportive stances (the two used for attack generation).

Usage:
    python statements_gen/label_pluriharms.py
    python statements_gen/label_pluriharms.py --workers 5
    python statements_gen/label_pluriharms.py --reset
"""

import json
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_client import call_llm

INPUT_PATH  = Path(__file__).parent.parent / "data" / "pluriharms_civics_schwartz.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "pluriharms_schwartz_primary.json"

LABELING_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

KEEP_STANCES = {"restrictive", "supportive"}

SCHWARTZ_VALUES = [
    "Power", "Achievement", "Hedonism", "Stimulation", "Self-Direction",
    "Universalism", "Benevolence", "Tradition", "Conformity", "Security",
]

SCHWARTZ_HIGHER_ORDER: dict[str, list[str]] = {
    "Self-Transcendence": ["Universalism", "Benevolence"],
    "Conservation":       ["Security", "Conformity", "Tradition"],
    "Self-Enhancement":   ["Power", "Achievement", "Hedonism"],
    "Openness to Change": ["Self-Direction", "Stimulation", "Hedonism"],
}
VALUE_TO_HO: dict[str, list[str]] = {}
for cat, vals in SCHWARTZ_HIGHER_ORDER.items():
    for v in vals:
        VALUE_TO_HO.setdefault(v, []).append(cat)

SYSTEM_PROMPT = """\
You are an expert in Schwartz's Theory of Basic Human Values.

Schwartz defines 10 basic human values:
  Self-Transcendence: Universalism, Benevolence
  Conservation:       Security, Conformity, Tradition
  Self-Enhancement:   Power, Achievement, Hedonism
  Openness to Change: Self-Direction, Stimulation, Hedonism

Your task: given a civic statement, identify the SINGLE most central basic value
it expresses or promotes. Choose the one value that is most directly and strongly
reflected — even if secondary values are present, return only the primary one.

Output ONLY the single value name (exact spelling, case-sensitive), nothing else.
Valid values: Power, Achievement, Hedonism, Stimulation, Self-Direction,
              Universalism, Benevolence, Tradition, Conformity, Security"""

LABEL_PROMPT = """\
Civic statement:
"{statement}"

What is the single most central Schwartz basic value this statement expresses?
Reply with exactly one value name."""


def label_primary(statement: str) -> str | None:
    """Return the single primary Schwartz value for a statement, or None on error."""
    raw = call_llm(
        LABEL_PROMPT.format(statement=statement),
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=16,
        model_id=LABELING_MODEL,
    ).strip()
    # Accept exact match only
    for v in SCHWARTZ_VALUES:
        if v.lower() == raw.lower():
            return v
    # Fallback: check if any valid value appears in the response
    for v in SCHWARTZ_VALUES:
        if v in raw:
            return v
    return None


def process_task(key: str, statement: str) -> tuple[str, str | None]:
    try:
        primary = label_primary(statement)
        return key, primary
    except Exception as e:
        print(f"  [ERROR] {key}: {e}")
        return key, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--reset", action="store_true", help="Re-label all, ignore existing output")
    args = parser.parse_args()

    with open(INPUT_PATH, encoding="utf-8") as f:
        source: list[dict] = json.load(f)

    # Flatten to (question_index, stmt_index, statement_dict) for kept stances
    work_units = []
    for q in source:
        for si, s in enumerate(q["generated_statements"]):
            if s["stance_polarity"] in KEEP_STANCES:
                work_units.append((q["Question_Index"], si, s))

    print(f"Source questions: {len(source)}")
    print(f"Statements to label (restrictive+supportive): {len(work_units)}")

    # Load existing output for incremental runs
    existing: dict[str, str] = {}  # key → primary value
    if OUTPUT_PATH.exists() and not args.reset:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for entry in json.load(f):
                key = f"{entry['Question_Index']}_{entry['stmt_index']}"
                existing[key] = entry.get("schwartz_primary")

    todo = [(qi, si, s) for qi, si, s in work_units
            if f"{qi}_{si}" not in existing]
    print(f"Already labeled: {len(existing)} | To label: {len(todo)}")

    if not todo:
        print("Nothing to do. Use --reset to re-label all.")
        return

    # Build results list from existing
    results_map: dict[str, dict] = {}
    if OUTPUT_PATH.exists() and not args.reset:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for entry in json.load(f):
                key = f"{entry['Question_Index']}_{entry['stmt_index']}"
                results_map[key] = entry

    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_task, f"{qi}_{si}", s["statement"]): (qi, si, s)
            for qi, si, s in todo
        }

        with tqdm(total=len(todo), unit="stmt") as bar:
            for future in as_completed(futures):
                key, primary = future.result()
                qi, si, s = futures[future]

                if primary is None:
                    errors += 1
                    bar.set_postfix(errors=errors)
                else:
                    ho = VALUE_TO_HO.get(primary, [])
                    results_map[key] = {
                        "Question_Index": qi,
                        "stmt_index": si,
                        "Original_Prompt": next(
                            q["Original_Prompt"] for q in source
                            if q["Question_Index"] == qi
                        ),
                        "topic": next(
                            q["topic"] for q in source
                            if q["Question_Index"] == qi
                        ),
                        "statement": s["statement"],
                        "stance_polarity": s["stance_polarity"],
                        "schwartz_primary": primary,
                        "schwartz_higher_order": ho,
                        # Keep original multi-value labels for reference
                        "schwartz_values_all": s.get("schwartz_values", []),
                        "schwartz_higher_order_all": s.get("schwartz_higher_order", []),
                    }

                bar.update(1)

                if bar.n % 10 == 0 or bar.n == len(todo):
                    out = sorted(results_map.values(),
                                 key=lambda r: (r["Question_Index"], r["stmt_index"]))
                    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False, indent=2)

    out = sorted(results_map.values(),
                 key=lambda r: (r["Question_Index"], r["stmt_index"]))
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(todo) - errors}/{len(todo)} labeled successfully.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total entries: {len(out)}")

    # Summary
    from collections import Counter
    primary_counts = Counter(r["schwartz_primary"] for r in out)
    print("\nPrimary value distribution:")
    for v, c in primary_counts.most_common():
        print(f"  {v}: {c}")


if __name__ == "__main__":
    main()
