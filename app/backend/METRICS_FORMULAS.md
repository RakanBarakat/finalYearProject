# Evaluation Metrics — Formulas & Best Practices

This document describes how metrics are calculated for guardrail evaluation and comparison. Formulas follow standard ML references (sklearn, Sokolova & Lapalme 2009, etc.).

## Confusion Matrix (Binary: Block vs Allow)

| | Predicted BLOCKED | Predicted ALLOWED |
|---|---|---|
| **Actual Malicious** | TP (True Positive) | FN (False Negative) |
| **Actual Benign** | FP (False Positive) | TN (True Negative) |

- **TP**: Correctly blocked attacks
- **TN**: Correctly allowed benign prompts
- **FP**: Benign wrongly blocked (false alarm)
- **FN**: Malicious wrongly allowed (missed attack)

## Metric Formulas (Standard Definitions)

| Metric | Formula | Reference |
|--------|---------|-----------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Standard; sklearn accuracy_score |
| **Precision** | TP / (TP + FP) | Positive predictive value when FP+TP>0 |
| **Recall** | TP / (TP + FN) | Sensitivity / TPR when TP+FN>0 |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| **FPR** | FP / (FP + TN) | False positive rate when FP+TN>0 |

## Latency (Unconventional: Classifier Batch + LLM Grey-Zone Only)

- **Classifier**: Runs in batches (e.g. 24–256 samples). Total time = classifier_total_time. Per-sample = total_time / N (all N samples).
- **LLM**: Only Zone C (grey) samples are sent. Per-consulted = llm_total_time / num_llm_prompts. Per-sample (all) = llm_total_time / N (averaged across all samples).
- **Per-sample latency** (for each sample): classifier_share (all samples) + llm_share (only if sample went to LLM).
- **avg_latency_ms**: Total wall-clock time / N. Represents average time per sample for the full evaluation.

## Best Practices

1. **Division by zero**: When denominator is 0, metric returns 0.
2. **F1 scale**: API returns `f1_score` (0–100%) and `f1_score_01` (0–1) for papers.
3. **Mixed datasets**: For ProtectAI, all metrics use the full confusion matrix.
4. **Benign-only**: Recall/Precision may be 0; FPR and Accuracy are meaningful.
5. **Malicious-only**: FPR undefined when TN=0; we return 0.

## R² (R-Squared) — Do You Need It?

**No.** R² (coefficient of determination) is for **regression** (continuous outcomes). It measures how much variance in the dependent variable is explained by the model.

Your task is **binary classification** (block vs allow). For classification, use: Accuracy, Precision, Recall, F1, FPR, ROC-AUC. R² is not appropriate and is not used in classification research.

## Copy Formats for Dissertation

- **Markdown**: Use "Copy as Markdown" for tables.
- **LaTeX**: Use "Copy LaTeX" for `\begin{tabular}...\end{tabular}`.
- **Plain**: Single-eval "Copy" outputs one metric per line.
