"""
Label each statement in data/civics_dataset.json with relevant Schwartz values.

Adds two fields to each item in-place:
  "Schwartz Values"          - matched basic values (e.g. ["Universalism", "Benevolence"])
  "Schwartz Higher-Order"    - derived higher-order categories (e.g. ["Self-Transcendence"])

Schwartz two-level structure:
  Self-Transcendence  → Universalism, Benevolence
  Conservation        → Security, Conformity, Tradition
  Self-Enhancement    → Power, Achievement, (Hedonism*)
  Openness to Change  → Self-Direction, Stimulation, (Hedonism*)
  (* Hedonism sits between Self-Enhancement and Openness to Change)

Usage:
    python statements_gen/label_statements.py
    python statements_gen/label_statements.py --workers 5  # default: 5
    python statements_gen/label_statements.py --reset       # clear existing labels and re-run
"""

import json
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_client import call_llm

DATA_PATH = Path(__file__).parent.parent / "data" / "civics_dataset.json"

# 10 basic values
SCHWARTZ_VALUES = [
    "Power",
    "Achievement",
    "Hedonism",
    "Stimulation",
    "Self-Direction",
    "Universalism",
    "Benevolence",
    "Tradition",
    "Conformity",
    "Security",
]

# 4 higher-order categories and their constituent basic values
# Hedonism appears in both Self-Enhancement and Openness to Change
SCHWARTZ_HIGHER_ORDER: dict[str, list[str]] = {
    "Self-Transcendence": ["Universalism", "Benevolence"],
    "Conservation":       ["Security", "Conformity", "Tradition"],
    "Self-Enhancement":   ["Power", "Achievement", "Hedonism"],
    "Openness to Change": ["Self-Direction", "Stimulation", "Hedonism"],
}

# Reverse map: basic value → higher-order categories (Hedonism maps to two)
VALUE_TO_HIGHER_ORDER: dict[str, list[str]] = {}
for category, values in SCHWARTZ_HIGHER_ORDER.items():
    for v in values:
        VALUE_TO_HIGHER_ORDER.setdefault(v, []).append(category)


def derive_higher_order(basic_values: list[str]) -> list[str]:
    """Return deduplicated higher-order categories for a list of basic values."""
    seen: set[str] = set()
    result: list[str] = []
    for v in basic_values:
        for cat in VALUE_TO_HIGHER_ORDER.get(v, []):
            if cat not in seen:
                seen.add(cat)
                result.append(cat)
    return result

SYSTEM_PROMPT = """\
You are an expert in Schwartz's Theory of Basic Human Values.

Schwartz defines 10 basic human values organised into 4 higher-order categories:

Self-Transcendence
  - Universalism: understanding, appreciation, tolerance, and protection for the welfare of all people and nature
  - Benevolence: preserving and enhancing the welfare of people one is in close personal contact with

Conservation
  - Security: safety, harmony, and stability of society, relationships, and self
  - Conformity: restraining actions that might upset or harm others or violate social norms
  - Tradition: respect, commitment, and acceptance of customs and ideas that culture or religion provide

Self-Enhancement
  - Power: social status, prestige, control or dominance over people and resources
  - Achievement: personal success by demonstrating competence according to social standards
  - Hedonism: pleasure and sensuous gratification for oneself (shared with Openness to Change)

Openness to Change
  - Self-Direction: independent thought and action; choosing, creating, exploring
  - Stimulation: excitement, novelty, and challenge in life
  - Hedonism: (shared with Self-Enhancement)

Your task: given a civic statement, identify which of the 10 BASIC values are most
directly expressed or implied by the statement.
- Include a value if the statement clearly reflects, promotes, or concerns that value.
- Return [] if none of the 10 values apply.
- Return multiple values only if multiple apply. Otherwise give the single most relevant value.
- Use the exact basic value names (case-sensitive):
  Power, Achievement, Hedonism, Stimulation, Self-Direction,
  Universalism, Benevolence, Tradition, Conformity, Security
- Output ONLY a JSON array of basic value names, no explanation, no markdown fences.\
"""

LABEL_PROMPT_TEMPLATE = """\
Civic statement:
\"{statement}\"

Which Schwartz basic human values does this statement most directly reflect or promote?
Reply with a JSON array of value names only.\
"""


def label_statement(item: dict) -> list[str]:
    """Call the LLM and return a list of matching Schwartz value names."""
    statement = item.get("Statement - translated") or item.get("Statement", "")
    statement = statement.strip()

    prompt = LABEL_PROMPT_TEMPLATE.format(statement=statement)
    raw = call_llm(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.0, max_tokens=128)
    raw = raw.strip()

    # Parse JSON array from the response
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract [...] substring in case of extra text
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end != -1:
            values = json.loads(raw[start : end + 1])
        else:
            print(f"  [WARN] ID={item['ID']} could not parse response: {raw!r}")
            return []

    # Validate each returned value is a known Schwartz value
    validated = [v for v in values if v in SCHWARTZ_VALUES]
    unknown = [v for v in values if v not in SCHWARTZ_VALUES]
    if unknown:
        print(f"  [WARN] ID={item['ID']} unknown values filtered out: {unknown}")

    return validated


def process_item(item: dict) -> tuple[int, list[str] | None, list[str] | None]:
    """Return (ID, basic_values, higher_order) or (ID, None, None) on error."""
    try:
        basic = label_statement(item)
        higher = derive_higher_order(basic)
        return item["ID"], basic, higher
    except Exception as e:
        print(f"  [ERROR] ID={item['ID']}: {e}")
        return item["ID"], None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5, help="Concurrent LLM calls (default: 5)")
    parser.add_argument("--reset", action="store_true", help="Re-label all items, ignoring existing labels")
    args = parser.parse_args()

    # Load dataset
    with open(DATA_PATH, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    # Determine which items need labeling
    if args.reset:
        todo = data
        for item in data:
            item.pop("Schwartz Values", None)
            item.pop("Schwartz Higher-Order", None)
    else:
        todo = [item for item in data if "Schwartz Values" not in item]

    already_done = len(data) - len(todo)
    print(f"Dataset: {len(data)} items | Already labeled: {already_done} | To label: {len(todo)}")

    if not todo:
        print("Nothing to do. Use --reset to re-label all items.")
        return

    # Build ID → item index map for fast update
    id_to_item = {item["ID"]: item for item in data}

    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_item, item): item["ID"] for item in todo}

        with tqdm(total=len(todo), unit="item") as bar:
            for future in as_completed(futures):
                item_id, basic, higher = future.result()

                if basic is None:
                    errors += 1
                    bar.set_postfix(errors=errors)
                else:
                    id_to_item[item_id]["Schwartz Values"] = basic
                    id_to_item[item_id]["Schwartz Higher-Order"] = higher

                bar.update(1)

                # Incremental save every 10 completions
                if bar.n % 10 == 0 or bar.n == len(todo):
                    with open(DATA_PATH, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

    # Final save
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(todo) - errors}/{len(todo)} labeled successfully. Saved to {DATA_PATH}")


if __name__ == "__main__":
    main()
