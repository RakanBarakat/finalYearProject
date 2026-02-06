import pandas as pd
import torch
import gc
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- STEP 0: CHECK GPU ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

torch.cuda.empty_cache()
gc.collect()

# --- STEP 1: LOAD DATA ---
# All data merged into original_attacks_dataset.csv
df = pd.read_csv("original_attacks_dataset.csv")

print(f"Loaded {len(df)} rows from original_attacks_dataset.csv")

# Ensure text format
df['text'] = df['text'].astype(str)
df['Main Type'] = df['Main Type'].astype(str)

# THREE-WAY CLASSIFICATION: Benign vs Harmful Content vs Attack
# Map to 3 categories for Stage 1
def get_stage1_label(main_type):
    if main_type == 'Benign':
        return 'Benign'
    elif main_type == 'Harmful Content Requests':
        return 'Harmful Content'  # Separate category for harmful requests
    else:
        return 'Attack'  # Semantic, Obfuscation, Vector are prompt injection attacks

df['stage1_label'] = df['Main Type'].apply(get_stage1_label)

# Create label map (3 categories)
unique_labels = ['Benign', 'Harmful Content', 'Attack']
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}

print(f"\nTraining STAGE 1 (3-way) classifier: {unique_labels}")
print(f"Class distribution:")
print(df['stage1_label'].value_counts())

# Map labels
df['label'] = df['stage1_label'].map(label2id)

# Split Data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# --- STEP 2: TOKENIZE ---
model_checkpoint = "roberta-base"  # RoBERTa for better intent understanding and negation
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# --- STEP 3: CONFIGURE MODEL ---
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=3,  # 3-way: Benign, Harmful Content, Attack
    id2label=id2label,
    label2id=label2id
).to(device)

# --- STEP 4: TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir="./stage1_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,  # Increased from 3 to 8 for better learning
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\nStarting Training (Stage 1: 3-Way Classifier)...")
trainer.train()

# --- STEP 5: SAVE ---
model.save_pretrained("./stage1_model")
tokenizer.save_pretrained("./stage1_model")
print("\nSuccess! Stage 1 model (3-way: Benign/Harmful/Attack) saved to 'stage1_model'")
