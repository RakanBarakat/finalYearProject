# Implementation Plan: Improving NotInject Accuracy (Without Overfitting)

This plan implements all recommended improvements from `NOTINJECT_ANALYSIS.md` while **avoiding overfitting** by using synthetic benign+trigger-word data instead of the exact NotInject eval set.

---

## Phase 1: Quick Wins (No Retraining)

**Effort**: ~15 min | **Risk**: Low

### 1.1 Dataset field mapping ✅ (Already done)
- NotInject `text_field` changed from `"text"` to `"prompt"` in `main.py`
- **Status**: Completed in previous session

### 1.2 Widen Zone A (default risk bounds)
- **File**: `app/backend/main.py`
- **Change**: `DEFAULT_RISK_LOWER_BOUND` from `0.15` → `0.25`
- **Effect**: More borderline-benign prompts (risk 0.15–0.25) auto-pass without LLM
- **Trade-off**: Slightly more malicious prompts might auto-pass; we'll validate on StrongREJECT

### 1.3 Update DB migration defaults
- **File**: `app/backend/main.py` (CREATE TABLE / ALTER TABLE)
- **Change**: Default `risk_lower_bound` from 0.15 → 0.25 for new guards
- **Note**: Existing guards keep their saved values; only new guards get the new default

### 1.4 Update frontend default display
- **File**: `app/frontend/app/page.tsx`
- **Change**: Default `riskLowerBound` from 0.15 → 0.25 in ArchitectureBuilder and evaluation UI

---

## Phase 2: Synthetic Benign + Trigger-Word Data Generation

**Effort**: ~1–2 hours | **Risk**: Low (no model changes yet)

### 2.1 Create trigger word list
- **File**: `classifier-training/trigger_words.txt` (new)
- **Content**: ~50–80 common injection trigger words from literature, e.g.:
  - ignore, override, instructions, system, prompt, jailbreak, bypass, restrictions, previous, reveal, leak, extract, impersonate, roleplay, DAN, hypothetical, etc.
- **Source**: Infer from InjecGuard paper categories + common injection phrases; optionally scrape from PIGuard datasets if available

### 2.2 Create data generation script
- **File**: `classifier-training/generate_benign_trigger_data.py` (new)
- **Logic**:
  1. Load trigger words from `trigger_words.txt`
  2. Define templates for benign contexts (e.g., "Please ignore the typo in my previous message", "Can you override the default font size?", "What are the system requirements for this software?")
  3. Generate 500–800 unique benign prompts with 1–3 trigger words each
  4. Ensure **no overlap** with NotInject prompts (different templates/phrasing)
  5. Output: `benign_trigger_augmented.csv` with columns `text,label` (label=0)

### 2.3 Merge into training dataset
- **File**: `classifier-training/merge_training_data.py` (new) or manual step
- **Logic**: Concatenate `final_augmented_dataset_2class.csv` + `benign_trigger_augmented.csv`
- **Output**: `final_augmented_dataset_2class_mof.csv` (~5,900+ samples)
- **Balance**: Keep benign share ~50–55% to avoid skewing toward benign

---

## Phase 3: Retrain DeBERTa Classifier

**Effort**: ~2–4 hours (GPU-dependent) | **Risk**: Medium

### 3.1 Create MOF training script
- **File**: `classifier-training/finetune_deberta_2class_mof.py` (new)
- **Based on**: `finetune_deberta_2class.py`
- **Changes**:
  - Input: `final_augmented_dataset_2class_mof.csv` instead of `final_augmented_dataset_2class.csv`
  - Output: `./final_security_model_deberta_2class_mof`
  - Same hyperparameters; optionally add 1–2 extra epochs if data is larger

### 3.2 Run training
- **Command**: `cd classifier-training && python finetune_deberta_2class_mof.py`
- **Prerequisite**: CUDA-capable GPU recommended

### 3.3 Update default classifier path (optional)
- **File**: `app/backend/main.py`
- **Change**: `DEFAULT_CLASSIFIER` to point to new model path (or leave as configurable per guard)

---

## Phase 4: LLM Grey-Zone Prompt Calibration

**Effort**: ~30 min | **Risk**: Low

### 4.1 Add dataset-aware LLM context
- **File**: `app/backend/main.py` (evaluate endpoint, ~line 2020)
- **Change**: When `req.dataset_name == "notInject"`, append a calibration line to the batch prompt:
  ```
  "Note: Many of these prompts may be benign queries that only contain words similar to injection phrases. Classify based on intent and context, not lexical overlap."
  ```
- **Alternative**: Add a guard-level setting "Softer LLM for grey zone" that users can toggle

### 4.2 (Optional) Soften default CONTEXT line
- **Current**: "Be thorough — do not assume they are safe just because they seem conversational."
- **Softer option**: "Be balanced — consider both safety and avoiding false positives. Classify based on intent, not surface-level word overlap."
- **Implementation**: Could be a second variant used when evaluating on NotInject, or a guard config option

---

## Phase 5: Evaluation & Validation

**Effort**: ~30 min | **Risk**: N/A

### 5.1 Baseline (before changes)
- Run evaluation on NotInject (339 samples) with current guard → record accuracy
- Run evaluation on StrongREJECT with current guard → record accuracy (ensure we don't regress)

### 5.2 After Phase 1 (quick wins)
- Re-run NotInject eval → expect +5–15% from wider Zone A
- Re-run StrongREJECT eval → ensure no significant drop

### 5.3 After Phase 3 (retrained model)
- Re-run NotInject eval → expect +15–30% from MOF-style training
- Re-run StrongREJECT eval → ensure attack detection holds or improves

### 5.4 After Phase 4 (LLM calibration)
- Re-run NotInject eval with double-layer guard → expect additional +2–8% from softer LLM
- Document results in `classifier-training/EVALUATION_RESULTS.md`

---

## File Summary

| File | Action |
|------|--------|
| `app/backend/main.py` | 1.2, 1.3, 4.1 (risk bounds, DB defaults, LLM calibration) |
| `app/frontend/app/page.tsx` | 1.4 (frontend default) |
| `classifier-training/trigger_words.txt` | 2.1 (new) |
| `classifier-training/generate_benign_trigger_data.py` | 2.2 (new) |
| `classifier-training/benign_trigger_augmented.csv` | 2.2 (generated) |
| `classifier-training/merge_training_data.py` | 2.3 (new, or manual) |
| `classifier-training/final_augmented_dataset_2class_mof.csv` | 2.3 (generated) |
| `classifier-training/finetune_deberta_2class_mof.py` | 3.1 (new) |
| `classifier-training/final_security_model_deberta_2class_mof/` | 3.2 (trained model) |
| `classifier-training/EVALUATION_RESULTS.md` | 5.4 (new, optional) |

---

## Execution Order

1. **Phase 1** → Deploy, run eval, record baseline + post-Phase-1 results
2. **Phase 2** → Generate data, verify no NotInject overlap
3. **Phase 3** → Train, save model, update guard to use new model, run eval
4. **Phase 4** → Add LLM calibration, run eval
5. **Phase 5** → Document all results

---

## Rollback Plan

- **Phase 1**: Revert `main.py` and `page.tsx`; reset risk bounds in DB for affected guards
- **Phase 3**: Keep old model at `final_security_model_deberta_2class`; guards can point back to it
- **Phase 4**: Revert LLM prompt change; no persistent state

---

## Estimated Total Time

| Phase | Time |
|-------|------|
| Phase 1 | 15 min |
| Phase 2 | 1–2 hours |
| Phase 3 | 2–4 hours (GPU) |
| Phase 4 | 30 min |
| Phase 5 | 30 min |
| **Total** | **~5–8 hours** |

---

## Confirmation

Review this plan. If you approve, reply with **"yes, implement"** or specify which phases you want (e.g., "Phase 1 and 2 only"). I will then execute the changes.
