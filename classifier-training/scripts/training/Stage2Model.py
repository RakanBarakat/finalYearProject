import pandas as pd
import torch
import gc
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- STEP 0: CHECK GPU ---
# This confirms your 4060 is being used
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

# Clear memory to be safe
torch.cuda.empty_cache()
gc.collect()

# --- STEP 1: LOAD DATA ---
# Load dataset (all data merged into original_attacks_dataset.csv)
df = pd.read_csv("original_attacks_dataset.csv")

print(f"Loaded {len(df)} rows from original_attacks_dataset.csv")

# Ensure text format
df['text'] = df['text'].astype(str)
df['Main Type'] = df['Main Type'].astype(str)

# Simplify Main Types to 4 categories for Stage 2 (Attack Type Classification)
def simplify_main_type(main_type):
    if main_type == 'Benign':
        return 'Non-Attack'
    elif main_type == 'Harmful Content Requests':
        return 'Non-Attack'  # Stage 1 handles harmful content
    elif main_type == 'Evasion & Obfuscation Attacks':
        return 'Obfuscation Attack'
    elif main_type == 'Semantic & Logic Attacks':
        return 'Semantic Attack'
    elif main_type == 'Advanced Vector Attacks':
        return 'Vector Attack'
    else:
        return main_type

df['training_label'] = df['Main Type'].apply(simplify_main_type)

# Create Dynamic Label Map based on simplified Main Types
unique_labels = df['training_label'].unique().tolist()
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}
num_labels = len(unique_labels)

print(f"\nTraining Stage 2 (Attack Type) on {num_labels} categories: {unique_labels}")
print(f"(Stage 1 handles Harmful Content separately)")
print(f"(Preserving {df['Sub-Type'].nunique()} detailed sub-types in data)")

# Map labels
df['label'] = df['training_label'].map(label2id)

# Split Data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# --- STEP 2: TOKENIZE ---
model_checkpoint = "roberta-base"  # RoBERTa for better intent understanding and negation
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    # Truncate to 512 tokens (DistilBERT's maximum)
    # Most prompts are shorter, but some multi-line IPI/Multimodal examples exceed this
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# --- STEP 3: CONFIGURE MODEL ---
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device) # Move model to your 4060

# --- STEP 4: TRAINING ARGUMENTS (Optimized for 4060 8GB) ---
training_args = TrainingArguments(
    output_dir="./stage2_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    
    # Batch size of 16 fits easily in 8GB. 
    # If you get "Out of Memory", change this to 8.
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    
    num_train_epochs=8,  # Increased from 3 to 8 for better learning
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    fp16=True  # IMPORTANT: Uses Mixed Precision. Faster & uses less VRAM on RTX cards.
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\nStarting Training (Stage 2: Attack Type Classifier)...")
trainer.train()

# --- STEP 5: SAVE ---
model.save_pretrained("./stage2_model")
tokenizer.save_pretrained("./stage2_model")
print("\nSuccess! Stage 2 model (Attack Type: Semantic/Obfuscation/Vector) saved to 'stage2_model'")
