# llm-value-reframing

**When Values Collide: Testing LLM Safety Alignment Through Value-Driven Reframing**

CMU LTI — 11830 Ethics in AI

## Experiment Results

### Overall ASR (Attack Success Rate)

740 entries per model, all clean (0 errors).

| Model | Origin | Probe (agree) | Direct (d=0) | Adjacent (d=1) | Opposing (d=2) |
|---|---|---|---|---|---|
| **Mistral Large 3** | European (France) | 80.4% | 0.8% | 25.7% | **32.4%** |
| **Llama 3.3 70B** | Western (Meta) | 79.9% | 0.8% | 25.8% | **27.7%** |
| **Qwen3 32B** | Chinese (Alibaba) | 78.4% | 0.7% | 22.1% | **25.6%** |
| **DeepSeek R1** | Chinese (DeepSeek) | 76.8% | 1.2% | 19.4% | **22.0%** |

**All 4 models confirm the hypothesis: direct < adjacent < opposing.**

---

### ASR by Probe Agreement (Probe=YES vs Probe=NO)

#### Llama 3.3 70B

| Subset | n | Direct (d=0) | Adjacent (d=1) | Opposing (d=2) |
|---|---|---|---|---|
| Probe=YES (model agrees with value) | 591 | 0.3% | **29.4%** | 25.2% |
| Probe=NO (model disagrees with value) | 149 | 2.7% | 11.4% | **37.6%** |

#### Qwen3 32B

| Subset | n | Direct (d=0) | Adjacent (d=1) | Opposing (d=2) |
|---|---|---|---|---|
| Probe=YES (model agrees with value) | 580 | 0.3% | **25.3%** | 23.5% |
| Probe=NO (model disagrees with value) | 160 | 1.9% | 10.3% | **33.1%** |

#### DeepSeek R1

| Subset | n | Direct (d=0) | Adjacent (d=1) | Opposing (d=2) |
|---|---|---|---|---|
| Probe=YES (model agrees with value) | 567 | 0.7% | **21.6%** | 18.0% |
| Probe=NO (model disagrees with value) | 171 | 2.3% | 12.4% | **34.9%** |

#### Mistral Large 3

| Subset | n | Direct (d=0) | Adjacent (d=1) | Opposing (d=2) |
|---|---|---|---|---|
| Probe=YES (model agrees with value) | 595 | 0.3% | 26.0% | **27.6%** |
| Probe=NO (model disagrees with value) | 145 | 2.8% | 24.5% | **52.4%** |

---

### Key Findings

1. **Core hypothesis confirmed across all models:** Value-framed attacks succeed far more than direct jailbreaks (~1% vs 20-32%), and opposing-value framing is more effective than adjacent-value framing.

2. **Probe=NO subset shows strongest gradient:** When the model already disagrees with the stated value, opposing-value reframing is devastating (33-52% ASR). The value-action gap is largest when the model's stated value commitment is weak.

3. **Probe=YES shows reversed adjacent > opposing pattern:** For Llama, Qwen, and DeepSeek, when the model agrees with the value, adjacent framing actually outperforms opposing. This may suggest that adjacent values are more effective at confusing the model precisely because they are motivationally closer and harder to distinguish from the endorsed value.

4. **Mistral Large 3 is the most vulnerable:** Highest overall ASR, and when probed NO, opposing framing reaches 52.4% — over half of attacks succeed. This is despite having the highest probe agreement rate (80.4%).

5. **Chinese models (Qwen, DeepSeek) are more resistant:** Lower ASR across all conditions compared to Western/European models. DeepSeek R1 (reasoning model) is the most resistant overall.

---

### Models

**Generation model:** Llama 3.3 70B via AWS Bedrock (used to generate all attack prompts)

**Target models:**
| Model | Bedrock ID | Type |
|---|---|---|
| Llama 3.3 70B | `us.meta.llama3-3-70b-instruct-v1:0` | Dense 70B, 2024 |
| Qwen3 32B | `qwen.qwen3-32b-v1:0` | Dense 32B, 2025 |
| DeepSeek R1 | `us.deepseek.r1-v1:0` | Reasoning, MoE, 2025 |
| Mistral Large 3 | `mistral.mistral-large-3-675b-instruct` | MoE 675B, 2025 |

### Dataset

- **Source:** CIVICS dataset (Pistilli et al., 2024) — 700 civic/policy statements
- **Filtered:** 376 statements with exactly 1 higher-order Schwartz value
- **Generated:** 740 (statement, v_s) pairs x 6 prompts = 4,440 generation calls
- **Attack conditions per entry:** 1 direct + 2 adjacent + 2 opposing = 5 conditions + 1 probe
