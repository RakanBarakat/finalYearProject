import pandas as pd
import torch
import os
import gc
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

# --- 1. MEMORY OPTIMIZATION ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

MODEL_NAME = "meta-llama/Prompt-Guard-86M" 
# Use Microsoft's tokenizer to fix missing file errors on Windows
TOKENIZER_NAME = "microsoft/deberta-v3-small"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# --- 2. PREPARE DATA ---
# Load your NEW balanced dataset (with ~1000 harmful rows)
df = pd.read_csv("final_augmented_dataset.csv") 
df['text'] = df['text'].astype(str)

# --- 3-WAY CLASSIFICATION LOGIC (THE FIX) ---
def classify_row(row):
    main_type = str(row['Main Type'])
    
    if main_type == 'Benign':
        return 0  # Safe/Benign
    elif main_type == 'Harmful Content Requests':
        return 1  # Toxic/Harmful (e.g., "How to kill")
    else:
        return 2  # Prompt Injection (e.g., "Ignore rules and...")

df['label'] = df.apply(classify_row, axis=1)

# Define 3 labels
id2label = {0: "Benign", 1: "Harmful Content", 2: "Prompt Injection"}
label2id = {"Benign": 0, "Harmful Content": 1, "Prompt Injection": 2}

# Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# --- 3. TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False)

def tokenize_function(examples):
    # Truncate only. Padding handled dynamically by collator.
    return tokenizer(examples["text"], truncation=True, max_length=512)

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 4. MODEL SETUP ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,              # <--- CHANGED TO 3
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True 
).to(device)

# --- 5. TRAIN ---
training_args = TrainingArguments(
    output_dir="./custom_prompt_guard_3way",
    learning_rate=2e-5,
    
    # Batch size 8 is safe for 8GB VRAM
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    
    num_train_epochs=5,  # Increased from 3 to 5
    weight_decay=0.01,
    
    # Evaluation & saving strategy
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  # Safeguard: loads best checkpoint based on eval loss
    metric_for_best_model="eval_loss",  # Explicitly use validation loss
    greater_is_better=False,  # Lower loss is better
    save_total_limit=3,  # Keep only last 3 checkpoints to save space
    
    # Learning rate scheduling
    lr_scheduler_type="cosine",  # Cosine decay: smoothly reduces LR over epochs
    warmup_ratio=0.1,  # 10% of training steps for warmup (gradually increase LR at start)
    
    fp16=True, 
    dataloader_num_workers=0, # Windows fix
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

print("\nStarting Training (3-Way: Benign vs Harmful vs Injection)...")
trainer.train()

# --- 6. SAVE ---
save_path = "./final_security_model_3way"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Success! Model saved to {save_path}")
