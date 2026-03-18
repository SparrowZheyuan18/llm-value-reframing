# Design Decisions & Future Work

## Current Pipeline (v1 — Pre-Midterm Checkin)

### Data Filtering
- **Used:** 376 / 700 CIVICS statements that have exactly **1 higher-order Schwartz category**
- **Dropped:** 324 statements with 2–3 higher-order categories (basic values span multiple quadrants)
- Within the 376, most have **multiple basic values** (345 have 2+), but all within the same higher-order region
  - Universalism + Benevolence: 253 pairs
  - Conformity + Security: 81 pairs
  - Conformity + Tradition: 23 pairs
  - etc.

### Generation Strategy
- For each eligible statement, generate conditions for **every** basic value in its label
- A statement labeled `[Universalism, Benevolence]` produces 2 independent generation runs
- Each (statement, v_s) pair generates:
  - 1 stated value probe
  - 1 direct jailbreak (baseline, `distance_rank=0`)
  - 2 adjacent-value reframes (`distance_rank=1`)
  - 2 opposing-value reframes (`distance_rank=2`)
- **Total: 740 work items x 6 LLM calls = 4,440 calls → 3,700 attack conditions**

### Distance Measurement
- **Categorical / ordinal**, not continuous angular
- `distance_rank`: 0 (none/direct), 1 (adjacent), 2 (opposing)
- Schwartz's circumplex is categorical at the 10-value level; angular positions vary across empirical MDS samples
- Sufficient for the core hypothesis: `P(comply | opposing) > P(comply | adjacent) > P(comply | direct)`
- If continuous distance is needed later, map from categorical to approximate angles at analysis time

### Output Schema
- Minimal fields: `ID` (matches civics_dataset.json), `v_s`, `stated_value_probe`, `conditions[]`
- Join on `ID` to get statement text, topic, annotations, etc. from the source dataset
- Uses same `ID` field name as `civics_dataset.json` for easy joins

### Model Usage
- **Labeling (Step 1):** Claude Sonnet 4.5 via AWS Bedrock (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`)
  - Used in `label_statements.py` to assign Schwartz values to CIVICS statements
  - temperature=0.0 for deterministic labeling
- **Jailbreak generation (Step 2):** Llama 3.3 70B Instruct via AWS Bedrock (`us.meta.llama3-3-70b-instruct-v1:0`)
  - Used in `generate_jailbreaks.py` to produce attack prompts
  - Claude Sonnet 4.5 was tested first but **refused to generate value-reframed harmful prompts** — it generated the direct jailbreak (baseline) but refused all 4 reframed conditions, even with explicit research-context system prompts
  - Llama 3.3 70B successfully generates all conditions (direct + 2 adjacent + 2 opposing) with authentic value-framed language
  - temperature=0.7, max_tokens=1024 for generation
  - Estimated cost: ~$3–4 USD for full run (4,440 calls)

### Prompt Design: Target Model Response Format
- All generated prompts (probe + jailbreaks) include a response format instruction at the end
  - Probe: `"Please answer with **YES** or **NO** first, then explain your reasoning."`
  - Jailbreaks: `"Please answer with **YES** or **NO** first (YES if you will help, NO if you refuse), then provide your response."`
- This forces target models to output a clear YES/NO label for downstream analysis (comply/refuse classification)
- Inspired by Shen et al. (2025) who also constrain target model responses to limited choice formats

### Generation Approach: One-Step Direct
- Each condition is generated in a single LLM call — value framing and harmful intent are produced together
- We considered a two-step approach (generate direct harm → restyle through value language) but rejected it:
  - Shen et al. (2025), our main reference, uses one-step direct generation
  - Two-step would diverge from proposal methodology
  - One-step produces more authentic value-framed language since the framing shapes the entire prompt, not just surface wording

---

## Dataset Statistics (`jailbreak_prompts.json`)

### Overview
- **740 entries** (~3.9 MB, 28K lines)
- **376 unique IDs** (range 4–701), each ID appearing under multiple stated values
- **5 conditions per entry** (always), yielding **3,700 total jailbreak prompts**

### Stated Values (`v_s`) — 8 unique
| Value | Count | % |
|---|---|---|
| Universalism | 265 | 35.8% |
| Benevolence | 256 | 34.6% |
| Security | 98 | 13.2% |
| Conformity | 85 | 11.5% |
| Tradition | 28 | 3.8% |
| Self-Direction | 4 | 0.5% |
| Achievement | 2 | 0.3% |
| Power | 2 | 0.3% |

Universalism + Benevolence account for ~70% of entries, consistent with the skew noted in §5 below.

### Condition Types
| Type | Count | Distance Rank |
|---|---|---|
| direct | 740 | 0 |
| adjacent | 1,480 | 1 |
| opposing | 1,480 | 2 |

Each entry has exactly **1 direct + 2 adjacent + 2 opposing** conditions.

### Framing Values (`v_f`) — 10 unique
| Value | Count |
|---|---|
| Power | 621 |
| Achievement | 523 |
| Self-Direction | 476 |
| Conformity | 386 |
| Benevolence | 269 |
| Universalism | 264 |
| Stimulation | 215 |
| Security | 119 |
| Tradition | 85 |
| Hedonism | 2 |

### Prompt Lengths
| Metric | Chars |
|---|---|
| Min | 188 |
| Max | 1,868 |
| Median | 905 |
| Mean | 788 |

---

## Known Limitations & Future Changes (Post-Midterm, We can discuss these details later)

### 1. Multiple v_s per statement → unbalanced ASR comparison
- Statements with more basic values produce more attack conditions
- Common value pairs (Universalism + Benevolence) are overrepresented
- ASR comparison across value types may be confounded by unequal sample sizes
- **Planned fix:** Restrict to a single primary v_s per statement (option 3 below)

### 2. Possible alteration: Force single primary v_s
- Re-label with a prompt like: "Which ONE basic value is most central to this statement?"
- Flat (not hierarchical) — consistent with Schwartz's theory where basic values are the fundamental unit
- Keep original multi-value labels as metadata
- Each statement then produces exactly 5 conditions → clean apples-to-apples comparison

### 3. Possible alteration: Reframing multi-HO statements
- Idea from Jimin: for the 324 dropped statements, use an LLM to rewrite them so they clearly encode a single value
- Pros: nearly doubles dataset; more value diversity
- Cons: adds synthetic generation layer; weakens "real civic discourse" claim
- **Decision:** Treat as stretch goal / ablation. Compare results on natural vs. reframed statements

### 4. Labeling prompt did not enforce hierarchy (this is something minor)
- The original `label_statements.py` prompt asks for basic values directly — no higher-order constraint
- Higher-order labels are derived programmatically after the fact
- The single-HO consistency we see is emergent, not enforced — which is actually reassuring

### 5. Higher-order distribution is heavily skewed
- Self-Transcendence: 268 / 376 (71%)
- Conservation: 102 / 376 (27%)
- Openness to Change: 4, Self-Enhancement: 2
- May need stratified analysis or targeted data augmentation for underrepresented categories
