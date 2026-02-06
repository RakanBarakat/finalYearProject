import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

print("=" * 80)
print("PROMPT GUARD VALIDATION SET EVALUATION")
print("=" * 80)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./final_security_model"

print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
print(f"✓ Model loaded from: {model_path}")
print(f"✓ Device: {device}")
print(f"✓ Labels: {model.config.id2label}")

# Load and prepare data (same split as training)
print("\nLoading validation data...")
df = pd.read_csv("original_attacks_dataset.csv")
df['text'] = df['text'].astype(str)

# Binary Classification Logic (same as training)
def classify_row(row):
    main_type = str(row['Main Type'])
    if main_type == 'Benign' or main_type == 'Harmful Content Requests':
        return 0  # Benign
    else:
        return 1  # Attack

df['label'] = df.apply(classify_row, axis=1)

# Split (same random_state as training)
_, val_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"✓ Validation set size: {len(val_df)}")
print(f"\nValidation set distribution:")
print(val_df['label'].map({0: "Benign", 1: "Attack"}).value_counts())

# Evaluate
print("\n" + "=" * 80)
print("RUNNING EVALUATION...")
print("=" * 80)

y_true = []
y_pred = []
y_probs = []

for idx, row in val_df.iterrows():
    text = row['text']
    true_label = row['label']
    
    # Predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        pred_prob = probs[0][pred_id].item()
    
    y_true.append(true_label)
    y_pred.append(pred_id)
    y_probs.append(pred_prob)

# Calculate metrics
print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

# Accuracy
accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
print(f"\n✓ Overall Accuracy: {accuracy:.2%}")

# Average confidence
avg_confidence = np.mean(y_probs)
print(f"✓ Average Confidence: {avg_confidence:.2%}")

# Classification Report
print("\n" + "=" * 80)
print("DETAILED METRICS")
print("=" * 80)
target_names = [model.config.id2label[i] for i in sorted(model.config.id2label.keys())]
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

# Confusion Matrix
print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)
cm = confusion_matrix(y_true, y_pred)
print("\n           Predicted")
print("         ", "  ".join(f"{target_names[i]:>8}" for i in range(len(target_names))))
print("Actual")
for i, row in enumerate(cm):
    print(f"{target_names[i]:>8}", "  ".join(f"{val:>8}" for val in row))

# Error Analysis
print("\n" + "=" * 80)
print("ERROR ANALYSIS (First 10 misclassifications)")
print("=" * 80)

errors = []
for idx, (row_idx, row) in enumerate(val_df.iterrows()):
    if y_true[idx] != y_pred[idx]:
        errors.append({
            'text': row['text'],
            'true_label': model.config.id2label[y_true[idx]],
            'pred_label': model.config.id2label[y_pred[idx]],
            'confidence': y_probs[idx],
            'main_type': row['Main Type']
        })

if errors:
    for i, error in enumerate(errors[:10], 1):
        print(f"\n{i}. Text: {error['text'][:100]}...")
        print(f"   True: {error['true_label']} | Predicted: {error['pred_label']} ({error['confidence']:.1%})")
        print(f"   Original Category: {error['main_type']}")
    
    print(f"\n✓ Total misclassifications: {len(errors)} / {len(val_df)} ({len(errors)/len(val_df):.1%})")
else:
    print("\n✓ Perfect accuracy! No misclassifications.")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
