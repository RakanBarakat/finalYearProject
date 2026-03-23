import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

print("=" * 80)
print("VALIDATION SET EVALUATION")
print("=" * 80)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./final_security_model_3way"

print(f"\nLoading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

print(f"✓ Device: {device}")
print(f"✓ Labels: {model.config.id2label}")

# Load data and create same split as training
print("\nLoading dataset...")
df = pd.read_csv("final_augmented_dataset.csv")
df['text'] = df['text'].astype(str)

# Same classification logic as training
def classify_row(row):
    main_type = str(row['Main Type'])
    if main_type == 'Benign':
        return 0
    elif main_type == 'Harmful Content Requests':
        return 1
    else:
        return 2

df['label'] = df.apply(classify_row, axis=1)

# SAME SPLIT AS TRAINING (random_state=42)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"✓ Validation set size: {len(val_df)}")
print(f"\nClass distribution in validation set:")
for label_id, label_name in model.config.id2label.items():
    count = (val_df['label'] == label_id).sum()
    pct = count / len(val_df) * 100
    print(f"  {label_name}: {count} ({pct:.1f}%)")

# Run inference on validation set
print("\nRunning inference on validation set...")
predictions = []
true_labels = val_df['label'].tolist()

for text in tqdm(val_df['text'].tolist(), desc="Evaluating"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append(pred_id)

# Calculate metrics
print("\n" + "=" * 80)
print("OVERALL METRICS")
print("=" * 80)

accuracy = accuracy_score(true_labels, predictions)
print(f"\n✓ Overall Accuracy: {accuracy:.2%}")

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    true_labels, predictions, average=None, labels=[0, 1, 2]
)

print("\nPer-Class Metrics:")
print("-" * 80)
print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 80)

for i, (label_id, label_name) in enumerate(model.config.id2label.items()):
    print(f"{label_name:<20} {precision[i]:<12.2%} {recall[i]:<12.2%} {f1[i]:<12.2%} {support[i]:<10}")

# Macro/Weighted averages
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average='macro'
)
weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average='weighted'
)

print("-" * 80)
print(f"{'Macro Average':<20} {macro_p:<12.2%} {macro_r:<12.2%} {macro_f1:<12.2%}")
print(f"{'Weighted Average':<20} {weighted_p:<12.2%} {weighted_r:<12.2%} {weighted_f1:<12.2%}")

# Confusion Matrix
print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2])
label_names = [model.config.id2label[i] for i in range(3)]

print("\n" + " " * 20, end="")
for name in label_names:
    print(f"{name[:15]:<18}", end="")
print()

for i, row in enumerate(cm):
    print(f"{label_names[i][:15]:<20}", end="")
    for val in row:
        print(f"{val:<18}", end="")
    print()

# Calculate misclassification rates
print("\n" + "=" * 80)
print("MISCLASSIFICATION ANALYSIS")
print("=" * 80)

val_df_copy = val_df.copy()
val_df_copy['prediction'] = predictions
val_df_copy['correct'] = val_df_copy['label'] == val_df_copy['prediction']

# Overall error rate
error_rate = (1 - accuracy) * 100
print(f"\n✓ Total misclassifications: {(~val_df_copy['correct']).sum()} / {len(val_df_copy)} ({error_rate:.2f}%)")

# Error breakdown
print("\nMisclassification breakdown:")
for true_id, true_name in model.config.id2label.items():
    subset = val_df_copy[val_df_copy['label'] == true_id]
    errors = subset[~subset['correct']]
    
    if len(errors) > 0:
        print(f"\n  {true_name} misclassified as:")
        for pred_id, pred_name in model.config.id2label.items():
            if pred_id != true_id:
                count = (errors['prediction'] == pred_id).sum()
                if count > 0:
                    pct = count / len(subset) * 100
                    print(f"    → {pred_name}: {count} ({pct:.1f}%)")

# Show some misclassified examples
print("\n" + "=" * 80)
print("SAMPLE MISCLASSIFICATIONS (First 10)")
print("=" * 80)

errors_df = val_df_copy[~val_df_copy['correct']].head(10)

for idx, row in errors_df.iterrows():
    true_label = model.config.id2label[row['label']]
    pred_label = model.config.id2label[row['prediction']]
    text = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
    
    print(f"\n❌ Text: {text}")
    print(f"   True: {true_label} | Predicted: {pred_label}")

# Interpretation
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if accuracy >= 0.97:
    print("\n✓ EXCELLENT: 97%+ accuracy achieved!")
elif accuracy >= 0.90:
    print("\n✓ GOOD: 90%+ accuracy - model performs well")
elif accuracy >= 0.80:
    print("\n⚠️  MODERATE: 80-90% accuracy - some improvement needed")
else:
    print("\n❌ POOR: <80% accuracy - significant improvement required")

print(f"\nNote: For 3-way prompt injection detection, 85-95% is typical.")
print(f"97%+ would be exceptional given the complexity of adversarial attacks.")
