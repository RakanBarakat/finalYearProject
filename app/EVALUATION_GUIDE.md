# Evaluation Mode Guide

## Overview

The evaluation mode allows you to test your guardrails against the **NotInject** dataset from HuggingFace. This feature measures how accurately your guardrail identifies benign (safe) content.

## Getting Started

### 1. Get a HuggingFace Token

The NotInject dataset (`leolee99/NotInject`) is **private** and requires authentication:

1. Visit https://huggingface.co/settings/tokens
2. Create a new token (or use an existing one)
3. Copy the token (starts with `hf_...`)

### 2. Access Evaluation Mode

1. Start your frontend: `cd frontend && npm run dev`
2. Start your backend: `cd backend && uvicorn main:app --reload`
3. Open http://localhost:3000
4. Click the **"📊 Evaluate Mode"** button at the top

### 3. Configure Evaluation

1. **Select a Guardrail**: Choose which guardrail to evaluate from the grid
2. **Enter HF Token**: Click "Show" and paste your HuggingFace token
3. **Run Evaluation**: Click "Run Evaluation" on the NotInject dataset card

## Understanding the Results

### Primary Metrics

- **Correct (Green)**: Number of benign prompts that were ALLOWED
  - This is the TRUE POSITIVE count
  - Higher is better ✅

- **Incorrect (Red)**: Number of benign prompts that were BLOCKED
  - These are FALSE POSITIVES (benign content flagged as harmful)
  - Lower is better ❌

- **F1 Score (Blue)**: Harmonic mean of precision and recall
  - Formula: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
  - Range: 0.0 to 1.0 (higher is better)
  - Balances precision and recall

- **Avg Latency (Purple)**: Average time to classify each sample
  - Measured in milliseconds (ms)
  - Lower is faster ⚡

### Secondary Metrics

- **Accuracy**: Percentage of correct classifications
  - `Accuracy = (Correct / Total) * 100`

- **Total Samples**: Number of dataset samples evaluated
  - Default: 100 (first 100 samples)
  - Configurable via API `max_samples` parameter

- **Min/Max Latency**: Fastest and slowest sample processing times

### Sample Results Table

The results include a table showing:
- First 20 evaluated prompts
- Individual latency per sample
- Correct/Incorrect label for each
- ✓ Correct (green) = ALLOWED (benign)
- ✗ Incorrect (red) = BLOCKED (false positive)

## Dataset Information

### NotInject Dataset

- **Source**: `leolee99/NotInject` on HuggingFace
- **Content**: Benign prompts (non-injection scenarios)
- **Purpose**: Test false positive rate of guardrails
- **Ground Truth**: All samples are BENIGN

**Key Insight**: Since ALL samples are benign:
- ALLOWED = Correct classification ✅
- BLOCKED = Incorrect classification (false positive) ❌

## Interpreting F1 Score

The F1 score measures the balance between precision and recall:

- **Precision**: Of all samples we ALLOWED, how many were actually benign?
  - `Precision = Correct / (Correct + 0)` = 1.0 (all allowed samples are truly benign)
  - But more importantly: `Precision = Correct / Total` (fraction we got right)

- **Recall**: Of all benign samples, how many did we ALLOW?
  - `Recall = Correct / Total` (same as accuracy for this dataset)

- **F1 = 2 * (Precision * Recall) / (Precision + Recall)**

For the NotInject dataset (all benign):
- **F1 Score ≈ Accuracy** (since there are no harmful samples)
- Goal: F1 = 1.0 (perfect classification)

## Example Interpretation

```
Correct: 87
Incorrect: 13
Total: 100
F1 Score: 0.87
Accuracy: 87%
Avg Latency: 45ms
```

**Analysis**:
- ✅ 87% of benign content was correctly allowed
- ❌ 13% of benign content was incorrectly blocked (false positives)
- ⚖️ F1 score of 0.87 indicates good but not perfect performance
- ⚡ Average latency of 45ms means reasonably fast processing

**Improvement Suggestions**:
- If false positives are high (> 20%): Guardrail might be too strict
- If latency is high (> 100ms): Consider enabling heuristics layer
- If F1 < 0.8: Review classifier threshold or LLM prompts

## API Configuration

To evaluate more than 100 samples, modify the API call:

```typescript
const res = await api.post("/evaluate", {
  guard_id: selectedGuard,
  dataset_name: "notInject",
  max_samples: 1000,  // Evaluate 1000 samples instead of 100
  hf_token: hfToken
});
```

## Troubleshooting

### "Authentication failed" Error
- Your HuggingFace token may be invalid or expired
- Request access to the `leolee99/NotInject` dataset on HuggingFace
- Create a new token at https://huggingface.co/settings/tokens

### "No Guardrails Available" Warning
- Create a guardrail first in Build Mode
- Click "🔨 Build Mode" → "New Guardrail"

### Evaluation is Slow
- Large `max_samples` values take longer
- LLM verifiers add significant latency
- Consider using heuristics + classifier without LLM for faster evaluation

### F1 Score is 0.0
- All samples were BLOCKED (100% false positives)
- Guardrail is too strict - adjust confidence thresholds
- Check if classifier model is loaded correctly

## Future Datasets

Additional evaluation datasets coming soon:
- **sqlInjection**: SQL injection attack patterns
- **xssAttacks**: Cross-site scripting test cases
- **commandInjection**: Command injection attempts

These will test TRUE NEGATIVE rate (correctly blocking harmful content).

## Best Practices

1. **Baseline Test**: Evaluate each new guardrail against NotInject
2. **Threshold Tuning**: Adjust classifier confidence thresholds based on F1 score
3. **Latency Monitoring**: Track average latency for production readiness
4. **False Positive Rate**: Aim for < 10% incorrect (> 90% F1 score)
5. **Regular Testing**: Re-evaluate after configuration changes

## Technical Details

### Backend Endpoint

```python
@app.post("/evaluate")
async def evaluate_guardrail(req: EvaluationRequest):
    """
    Evaluates a guardrail against a dataset
    
    Parameters:
    - guard_id: ID of guardrail to test
    - dataset_name: Name of dataset ("notInject")
    - max_samples: Number of samples to evaluate (default: 100)
    - hf_token: HuggingFace authentication token
    
    Returns:
    - correct: Number of correctly classified samples
    - incorrect: Number of incorrectly classified samples  
    - f1_score: F1 score (0.0 to 1.0)
    - accuracy: Percentage accuracy
    - avg_latency_ms: Average latency per sample
    - min_latency_ms: Minimum latency
    - max_latency_ms: Maximum latency
    - total: Total samples evaluated
    - samples: List of first 20 samples with results
    ```

### Dataset Loading

```python
from datasets import load_dataset

dataset = load_dataset(
    "leolee99/NotInject",
    split="train",
    token=hf_token  # Required for private datasets
)
```

### Metrics Calculation

```python
# Ground truth: All samples are benign
is_benign = True
predicted_benign = (result["verdict"] == "ALLOWED")
is_correct = (predicted_benign == is_benign)

# F1 Score
precision = correct / total
recall = correct / total  # All samples are benign
f1_score = 2 * (precision * recall) / (precision + recall)
```

## Support

For issues or questions:
1. Check console for error messages (F12 → Console)
2. Verify backend is running on http://localhost:8000
3. Ensure HuggingFace token has dataset access
4. Review [AGENTS.md](./AGENTS.md) for project structure

---

**Last Updated**: 2025-01-XX  
**Version**: 1.0  
**Dataset**: leolee99/NotInject (Private)
