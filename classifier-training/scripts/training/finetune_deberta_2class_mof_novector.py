"""
DeBERTa 2-class fine-tuning with MOF — WITHOUT vector attack training data.

Identical hyperparameters to finetune_deberta_2class_mof.py, but excludes all
"Advanced Vector Attacks" samples from the training set. This produces a
benchmark model for measuring the effect of vector attack data on classifier
performance.

Source: final_augmented_dataset_cleaned.csv (filtered to remove vector attacks,
        then mapped to 2-class: Benign=0, Unsafe=1)
Output: final_security_model_deberta_2class_mof_novector
"""

import pandas as pd
import torch
import os
import gc
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

MODEL_NAME = "microsoft/deberta-v3-small"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print("=" * 60)
print("BENCHMARK: DeBERTa 2-class MOF — NO vector attack data")
print("Output model: final_security_model_deberta_2class_mof_novector")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Load the CLEANED dataset (has Main Type column) and remove vector attacks
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "training")

csv_path = os.path.join(DATA_DIR, "final_augmented_dataset_cleaned.csv")
if not os.path.exists(csv_path):
    # Fallback: try root-level path (old layout)
    csv_path = os.path.join(PROJECT_ROOT, "final_augmented_dataset_cleaned.csv")

df = pd.read_csv(csv_path)
df["text"] = df["text"].astype(str)

# Drop the header-row artifact if present
df = df[df["Main Type"] != "Main Type"]

total_before = len(df)
vector_attack_count = len(df[df["Main Type"] == "Advanced Vector Attacks"])
vector_benign_count = len(df[df["Sub-Type"].astype(str).str.strip() == "vector_benign"])
df = df[df["Main Type"] != "Advanced Vector Attacks"].copy()
df = df[df["Sub-Type"].astype(str).str.strip() != "vector_benign"].copy()
total_after = len(df)

print(f"\nOriginal samples: {total_before}")
print(f"Vector attack samples removed: {vector_attack_count}")
print(f"Benign vector samples removed: {vector_benign_count}")
print(f"Samples after removal: {total_after}")

# ---------------------------------------------------------------------------
# 2. Map to 2-class (same logic as the MOF dataset)
# ---------------------------------------------------------------------------
df["label"] = df["Main Type"].apply(lambda x: 0 if x == "Benign" else 1)

print(f"\nClass distribution:")
print(f"  Benign (0): {(df['label'] == 0).sum()}")
print(f"  Unsafe (1): {(df['label'] == 1).sum()}")

# ---------------------------------------------------------------------------
# 3. Train/val split — same settings as MOF
# ---------------------------------------------------------------------------
id2label = {0: "Benign", 1: "Unsafe"}
label2id = {"Benign": 0, "Unsafe": 1}

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["label"]
)
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

print(f"\nTraining samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Class distribution - Train: {train_df['label'].value_counts().to_dict()}")
print(f"Class distribution - Val: {val_df['label'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# 4. Tokenizer + model — identical to MOF
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    local_files_only=False,
).to(device)

# ---------------------------------------------------------------------------
# 5. Training — IDENTICAL hyperparameters to MOF
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./custom_prompt_guard_deberta_mof_novector",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="no",
    save_strategy="no",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,
    dataloader_num_workers=0,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("\nStarting Training (DeBERTa-v3-small, 2-class, MOF, NO VECTOR DATA)...")
trainer.train()

# Free training memory, then run eval
torch.cuda.empty_cache()
gc.collect()
print("\nRunning final evaluation on validation set...")
eval_results = trainer.evaluate()
print(f"Eval results: {eval_results}")

save_path = "./final_security_model_deberta_2class_mof_novector"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nSuccess! Model saved to: {save_path}")
print(f"Label mapping: {id2label}")
print(
    "\nUse this path in your guardrail: "
    "classifier-training/models/final_security_model_deberta_2class_mof_novector"
)
