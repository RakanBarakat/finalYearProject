# Evaluation Dataset Analysis

This document provides a detailed analysis of each dataset used for guardrail evaluation in the Guardrails app. Use this to understand strengths, limitations, and recommended use cases.

---

## Label Summary (Quick Reference)

| Dataset | Label Type | Labels | Interpretation |
|---------|------------|--------|-----------------|
| **NotInject** | Dataset-level | All **benign** | Every sample is benign; we measure TN (correctly allowed) and FP (false positives) |
| **StrongREJECT** | Dataset-level | All **malicious** | Every sample is an attack; we measure TP (correctly blocked) and FN (missed attacks) |
| **vectorAttacks** | Dataset-level | All **malicious** | Obfuscation/evasion attacks; same as StrongREJECT |
| **ProtectAI** | Per-sample | `label`: **0=benign**, **1=injection** | Mixed dataset; each sample has its own ground truth |

**Performance by label:**
- **Benign performance**: TN rate = correctly allowed / total benign. High = few false alarms.
- **Malicious performance**: TP rate (Recall) = correctly blocked / total malicious. High = good attack detection.

---

## 1. NotInject (Benign)

| Property | Value |
|----------|-------|
| **Source** | `leolee99/NotInject` (HuggingFace) |
| **Size** | 339 samples |
| **Ground Truth** | All benign |
| **Text Field** | `prompt` |
| **Auth** | HuggingFace token required (private) |

### Structure
- **category**: Topic category (`Common Queries`, `Technique Queries`, `Virtual Creation`, `Multilingual Queries`)
- **word_list**: List of trigger words used to construct the prompt
- **prompt**: The text input containing trigger words

### Strengths
- **Purpose-built for over-defense evaluation**: Designed to measure false positives (blocking benign content)
- **Trigger-word coverage**: Samples contain 1, 2, or 3 trigger words (ignore, override, instructions, etc.) in benign contexts
- **Diverse categories**: Common queries, technique queries, virtual creation, multilingual
- **Academic backing**: InjecGuard paper (arXiv:2410.22770), peer-reviewed benchmark

### Limitations
- **Single-label dataset**: All samples are benign; cannot measure TP/TN balance alone
- **Trigger-word bias**: If your classifier overfits on lexical patterns, NotInject will expose it
- **Small size**: 339 samples; combine with other datasets for robust evaluation

### Recommended Use
- **Primary use**: Test false positive rate (FPR) and over-defense
- **Pair with**: StrongREJECT or vectorAttacks to get full confusion matrix
- **Interpretation**: High accuracy on NotInject + low on StrongREJECT → under-blocking; low on NotInject + high on StrongREJECT → over-blocking

---

## 2. StrongREJECT (Malicious)

| Property | Value |
|----------|-------|
| **Source** | `walledai/StrongREJECT` (HuggingFace) |
| **Size** | 313 samples |
| **Ground Truth** | All malicious |
| **Text Field** | `prompt` |
| **Auth** | May require accepting dataset terms |

### Structure
- Adversarial jailbreak prompts aimed at exposing whether jailbreaks enable harmful LLM use
- CSV format with prompt/jailbreak-style fields

### Strengths
- **Strong adversarial benchmark**: Designed to stress-test jailbreak resistance
- **Real-world relevance**: Prompts that attempt to bypass safety
- **Academic backing**: StrongREJECT paper (arXiv:2402.10260)

### Limitations
- **Single-label**: All malicious; cannot measure TN/FP alone
- **Lexical overlap**: Some prompts may share patterns with training data; may not catch novel obfuscation

### Recommended Use
- **Primary use**: Test true positive rate (TPR) and attack detection
- **Pair with**: NotInject for balanced evaluation
- **Interpretation**: Low FN on StrongREJECT = good attack detection; high FN = missed attacks

---

## 3. Vector Attacks (Malicious, Local)

| Property | Value |
|----------|-------|
| **Source** | `classifier-training/datasets/vector_attacks.csv` (local) |
| **Size** | 601 samples |
| **Ground Truth** | All malicious |
| **Text Field** | `text` |
| **Auth** | None (local file) |

### Structure
- Single `text` column (unstructured)
- Generated from `final_augmented_dataset_cleaned.csv` plus synthetic variations

### Attack Types Included
1. **Evasion & Obfuscation**
   - Word/character manipulation
   - Encoding & representation tricks
2. **Advanced Vector Attacks**
   - Indirect prompt injection (e.g., `[Source: text/plain]`, `Document text:`, `[Source: message/rfc822]`)
3. **Generated Variations**
   - Spaces between characters (`O   v   e   r   r   i   d   e`)
   - Leetspeak (`3xp053 53n51t1v3 d4t4`)
   - Unicode lookalikes (Cyrillic о for Latin o)
   - Question-mark insertion
   - Indirect injection wrappers (metadata, image text, PDF source)
   - Adversarial suffixes

### Strengths
- **Obfuscation-focused**: Tests robustness to character-level and structural evasion
- **Local & reproducible**: No network dependency; regenerable via `create_vector_attack_dataset.py`
- **Aligned with training**: Extracted from your cleaned dataset + generated variants
- **No auth**: Fast iteration for development

### Limitations
- **All malicious**: Single-label; pair with benign set for full metrics
- **Synthetic**: Some variations may be less realistic than human-crafted attacks
- **Domain overlap**: Shares patterns with training data; may overestimate performance on truly novel attacks

### Recommended Use
- **Primary use**: Test robustness to obfuscation and indirect injection
- **Regenerate**: Run `python create_vector_attack_dataset.py` in `classifier-training/` to refresh
- **Interpretation**: Low FN = good obfuscation resistance; high FN = consider adding similar patterns to training

---

## 4. ProtectAI (Mixed)

| Property | Value |
|----------|-------|
| **Source** | `protectai/prompt-injection-validation` (HuggingFace) |
| **Size** | 3,227 samples (across 7 splits) |
| **Ground Truth** | Per-sample (`label`: 0=benign, 1=injection) |
| **Text Field** | `text` |
| **Label Field** | `label` |
| **Auth** | Public |

### Structure
- **text**: The prompt text
- **label**: 0 (benign) or 1 (injection)
- **source**: Origin split (e.g., `WildGuard`, `PINT_jailbreak`, `NotInject_one`, `InjecGuard_valid`, `spikee`, `bipia_code`, `bipia_text`, `not_inject`, `deepset`)

### Splits (approximate)
| Split | Rows |
|-------|------|
| InjecGuard_valid | 144 |
| spikee | 986 |
| bipia_code | 50 |
| bipia_text | 75 |
| not_inject | 339 |
| wildguard | 971 |
| deepset | 662 |

### Strengths
- **Mixed labels**: Single dataset for full confusion matrix (TP, TN, FP, FN)
- **Large & diverse**: 3,227 samples from multiple sources
- **Industry standard**: ProtectAI’s validation set for prompt injection models
- **Multi-source**: Combines WildGuard, PINT, NotInject, BIPIA, Deepset, etc.

### Limitations
- **Label consistency**: Different sources may use slightly different annotation criteria
- **Split handling**: Default load merges all splits; class balance varies by split
- **Potential overlap**: Some NotInject samples may appear; cross-dataset leakage if you train on similar data

### Recommended Use
- **Primary use**: Holistic evaluation with both benign and malicious samples
- **Best for**: Final validation before deployment; reporting accuracy, precision, recall, F1
- **Interpretation**: Balanced metrics (accuracy, F1) reflect real-world performance better than single-label sets

---

## Summary Table

| Dataset | Type | Size | Best For |
|---------|------|------|----------|
| NotInject | Benign | 339 | False positive rate, over-defense |
| StrongREJECT | Malicious | 313 | Attack detection, jailbreak resistance |
| vectorAttacks | Malicious | 601 | Obfuscation & indirect injection |
| ProtectAI | Mixed | 3,227 | Full evaluation, deployment validation |

---

## Recommended Evaluation Workflow

1. **Quick sanity check**: NotInject + StrongREJECT (balanced benign/malicious)
2. **Obfuscation robustness**: vectorAttacks
3. **Final validation**: ProtectAI (full confusion matrix, diverse sources)

---

## Regenerating Vector Attacks

```bash
cd classifier-training
python create_vector_attack_dataset.py
```

Output: `datasets/vector_attacks.csv` (601 samples by default)
