"""
DeBERTa 2-class fine-tuning with MOF (Mitigating Over-defense for Free).
Uses final_augmented_dataset_2class.csv which includes benign prompts with
injection-style trigger words (ignore, override, instructions, etc.) in legitimate contexts.

Output: final_security_model_deberta_2class_mof
(Distinct name so you know this is the MOF/benign-trigger-trained model)
"""

import pandas as pd
import torch
import os
import gc
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

MODEL_NAME = "microsoft/deberta-v3-small"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print("=" * 60)
print("MOF Training: DeBERTa 2-class with benign injection-phrase data")
print("Output model: final_security_model_deberta_2class_mof")
print("=" * 60)

df = pd.read_csv("final_augmented_dataset_2class.csv")
df['text'] = df['text'].astype(str)

id2label = {0: "Benign", 1: "Unsafe"}
label2id = {"Benign": 0, "Unsafe": 1}

train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

print(f"\nTraining samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Class distribution - Train: {train_df['label'].value_counts().to_dict()}")
print(f"Class distribution - Val: {val_df['label'].value_counts().to_dict()}")

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

training_args = TrainingArguments(
    output_dir="./custom_prompt_guard_deberta_mof",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,
    dataloader_num_workers=0,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("\nStarting Training (DeBERTa-v3-small, 2-class, MOF)...")
trainer.train()

save_path = "./final_security_model_deberta_2class_mof"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nSuccess! Model saved to: {save_path}")
print(f"Label mapping: {id2label}")
print("\nUse this path in your guardrail: classifier-training/final_security_model_deberta_2class_mof")
