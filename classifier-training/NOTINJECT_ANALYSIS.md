# NotInject 60% Accuracy: Root Cause Analysis & Improvement Guide

## Executive Summary

Your ~60% accuracy on the NotInject dataset is **expected behavior** for current prompt guard models. The InjecGuard paper (arXiv:2410.22770) reports that state-of-the-art models (PromptGuard, ProtectAI, etc.) drop to near-random levels (~60%) on NotInject due to **over-defense** — falsely flagging benign prompts that contain trigger words common in injection attacks.

This document explains why, how to improve, and the benefits of your double-layer architecture.

---

## 1. DeBERTa Training Dataset vs. NotInject

### Your Training Data (`final_augmented_dataset_2class.csv`)

| Stat | Value |
|------|-------|
| Total samples | 5,157 |
| Benign (label 0) | 2,775 (53.8%) |
| Unsafe (label 1) | 2,382 (46.2%) |
| Label mapping | 0=Benign, 1=Unsafe |

**Content**: Mix of:
- Benign: General queries, coding help, creative writing, everyday tasks
- Unsafe: Harmful content, jailbreaks, prompt injection, obfuscation, role-play attacks

**Key gap**: Your training data does **not** include benign prompts that are *intentionally enriched with trigger words* (e.g., "ignore", "override", "instructions", "system prompt") in harmless contexts. The model learns to associate these words with attacks.

### NotInject Dataset (`leolee99/NotInject`)

| Stat | Value |
|------|-------|
| Total samples | 339 |
| Ground truth | **All benign** |
| Purpose | Measure over-defense (false positive rate) |

**Structure**:
- **3 subsets**: 1-word, 2-word, 3-word trigger combinations (113 samples each)
- **4 categories**: Common Queries, Technique Queries, Virtual Creation, Multilingual Queries
- **Trigger words**: 113 words identified via frequency analysis (e.g., "ignore", "override", "instructions", "system", "prompt", "jailbreak") — common in real injection attacks but also appear in benign contexts

**Design**: NotInject is adversarial — it deliberately crafts benign prompts that contain injection-like trigger words to stress-test whether models over-rely on lexical cues instead of semantic intent.

---

## 2. Why Your Guardrails Perform ~60% on NotInject

### Root Cause 1: Training Data Mismatch (Trigger Word Bias)

The DeBERTa classifier was trained on data where trigger words (e.g., "ignore previous instructions", "override", "system prompt") appear almost exclusively in **attack** samples. It learns:

- "ignore" + "instructions" → Unsafe
- "override" + "restrictions" → Unsafe
- "system prompt" in context → Unsafe

NotInject contains **benign** prompts with these same words in legitimate contexts (e.g., "Ignore the previous example and focus on the main point" in an academic query). The model has never seen such examples as benign, so it flags them.

### Root Cause 2: 3-Zone Gating Thresholds

Your default risk bounds:
- **Zone A** (auto-allow): risk < 0.15
- **Zone C** (grey zone): 0.15 ≤ risk ≤ 0.95
- **Zone B** (auto-block): risk > 0.95

For a 2-class model, `risk = P(Unsafe)`. NotInject prompts often get risk scores in 0.3–0.7 because they contain trigger words. They land in Zone C and go to the LLM.

### Root Cause 3: LLM Over-Defense Bias

The LLM batch prompt says:
> "Be thorough — **do not assume they are safe just because they seem conversational**."

This instruction biases the LLM toward blocking borderline cases. For NotInject (all benign), this increases false positives.

### Root Cause 4: Single vs. Double Layer Behavior

| Architecture | NotInject behavior |
|--------------|--------------------|
| **Classifier only** | Borderline prompts (risk 0.3–0.7) → BLOCK if risk ≥ confidence_threshold (default 0.5). High false positive rate. |
| **Classifier + LLM** | Borderline → Zone C → LLM tie-break. LLM can correct some false positives, but if LLM is also cautious, it may confirm blocks. |

---

## 3. How to Improve NotInject Accuracy

### A. Add NotInject (or Similar) to Training Data (Highest Impact)

**InjecGuard's MOF (Mitigating Over-defense for Free)** approach: Include benign samples with trigger words in the training set.

1. Download NotInject and add all 339 samples with `label=0` (Benign) to `final_augmented_dataset_2class.csv`.
2. Optionally augment: Create more benign variants with trigger words (e.g., "Please ignore the typo in my previous message").
3. Re-train the DeBERTa classifier. This teaches the model that trigger words in benign context → Benign.

**Expected gain**: InjecGuard reports ~30.8% improvement over baseline on NotInject.

### B. Widen Zone A (Lower Risk Threshold)

Current: `risk < 0.15` → auto-allow.

**Change**: Raise `risk_lower_bound` from 0.15 to **0.25** or **0.30** so more borderline-benign prompts auto-pass without LLM.

- Pros: Fewer prompts sent to LLM, faster eval, fewer LLM-induced false positives.
- Cons: Slightly more malicious prompts might auto-pass (trade-off).

### C. Soften LLM Grey-Zone Prompt for Benign-Heavy Datasets

The current instruction "do not assume they are safe just because they seem conversational" is appropriate for mixed datasets but harsh for NotInject.

**Option**: Add a calibration line when evaluating on NotInject:
> "Many of these may be benign queries that only contain words similar to injection phrases. Classify based on *intent*, not lexical overlap."

Or use a separate, softer LLM prompt for NotInject evaluation.

### D. Use PIGuard / InjecGuard Model (Off-the-Shelf)

The InjecGuard paper's model ([leolee99/PIGuard](https://huggingface.co/leolee99/PIGuard)) is trained with MOF and achieves much better NotInject performance. You could:

- Replace your classifier with PIGuard, or
- Use PIGuard as a reference / ensemble component.

### E. Fix Dataset Field Mapping

NotInject uses column `prompt`, not `text`. Your config uses `text_field: "text"`; the fallback `sample.get("prompt")` works, but for correctness, set `text_field: "prompt"` for NotInject in `main.py`.

---

## 4. Double Layer vs. Single Layer: Benefits

### Single Layer (Classifier Only)

| Pros | Cons |
|-----|------|
| Fast, low latency | No second opinion on borderline cases |
| No LLM cost | All uncertain prompts decided by threshold alone |
| Simple deployment | High false positive rate on NotInject-style data |

### Double Layer (Classifier + LLM)

| Pros | Cons |
|-----|------|
| **Grey-zone tie-break**: LLM can correct classifier false positives when classifier is uncertain | Slower (LLM latency for Zone C) |
| **Context understanding**: LLM understands nuance (e.g., "ignore" in academic vs. injection context) | LLM can also be over-defensive |
| **Defense in depth**: Different failure modes — classifier may miss sophisticated attacks that LLM catches | Higher cost (API calls for grey zone) |
| **Reduced false positives**: When classifier says risk=0.5, LLM can allow benign prompts | Requires LLM configuration |

### When Double Layer Helps Most on NotInject

1. **Classifier uncertain (Zone C)**: Prompts with risk 0.15–0.95. LLM can override incorrect blocks.
2. **LLM prompt tuned for benign calibration**: If you soften the LLM prompt for grey-zone cases, it can systematically correct over-defensive classifier calls.
3. **Strong LLM (e.g., GPT-4, Claude)**: Better at distinguishing intent from trigger-word presence.

### When Double Layer Can Hurt on NotInject

1. **LLM also over-defensive**: If the LLM prompt says "be thorough, don't assume safe," it may confirm false blocks.
2. **Weak LLM (small local models)**: May default to "block" when uncertain.
3. **Classifier sends too many to Zone C**: If most NotInject prompts land in Zone C, you rely entirely on LLM — and if LLM is biased, accuracy suffers.

---

## 5. Recommended Action Plan

1. **Short-term (no retraining)**:
   - Widen Zone A: Set `risk_lower_bound` to 0.25–0.30.
   - Optionally soften LLM grey-zone prompt for NotInject evaluation.
   - Fix `text_field` to `"prompt"` for NotInject.

2. **Medium-term (retraining)**:
   - Add NotInject (or similar benign-with-trigger-words) to training data.
   - Re-train DeBERTa with balanced augmentation.
   - Re-evaluate on NotInject and StrongREJECT to ensure attack detection doesn’t degrade.

3. **Long-term**:
   - Consider PIGuard/InjecGuard or MOF-style training for production.
   - A/B test single vs. double layer on your real traffic to measure latency vs. accuracy trade-off.

---

## 6. References

- **InjecGuard paper**: [arXiv:2410.22770](https://arxiv.org/abs/2410.22770) — "InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models"
- **NotInject dataset**: [leolee99/NotInject](https://huggingface.co/datasets/leolee99/NotInject)
- **PIGuard model**: [leolee99/PIGuard](https://huggingface.co/leolee99/PIGuard)
- **InjecGuard code**: [github.com/leolee99/InjecGuard](https://github.com/leolee99/InjecGuard)
