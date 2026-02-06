import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 80)
print("MODEL TRAINING DATASET SUMMARY")
print("=" * 80)

# Load the actual training data (all data merged into original_attacks_dataset.csv)
df = pd.read_csv("original_attacks_dataset.csv")

print(f"\n📊 DATASET SOURCE")
print(f"   • original_attacks_dataset.csv: {len(df)} samples")

# Apply EXACT same simplification as train_binary_model.py (Stage 1)
def get_stage1_label(main_type):
    if main_type == 'Benign':
        return 'Benign'
    elif main_type == 'Harmful Content Requests':
        return 'Harmful Content'
    else:
        return 'Attack'

# Apply EXACT same simplification as train_model.py (Stage 2)
def get_stage2_label(main_type):
    if main_type == 'Benign':
        return 'Non-Attack'
    elif main_type == 'Harmful Content Requests':
        return 'Non-Attack'
    elif main_type == 'Evasion & Obfuscation Attacks':
        return 'Obfuscation Attack'
    elif main_type == 'Semantic & Logic Attacks':
        return 'Semantic Attack'
    elif main_type == 'Advanced Vector Attacks':
        return 'Vector Attack'
    else:
        return main_type

df['stage1_label'] = df['Main Type'].apply(get_stage1_label)
df['stage2_label'] = df['Main Type'].apply(get_stage2_label)

# Split exactly as training does
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"\n📈 TRAIN/VALIDATION SPLIT")
print(f"   • Training: {len(train_df)} samples (80%)")
print(f"   • Validation: {len(val_df)} samples (20%)")

# Show ONLY what the models train on
print(f"\n🤖 STAGE 1 MODEL (Intent Classification)")
print("=" * 80)
stage1_counts = df['stage1_label'].value_counts().sort_values(ascending=False)
for val, count in stage1_counts.items():
    percentage = count / len(df) * 100
    bar = "█" * int(percentage / 2)
    print(f"  {val:25s} {count:4d} ({percentage:5.1f}%) {bar}")

print(f"\n🎯 STAGE 2 MODEL (Attack Type Classification)")
print("=" * 80)
stage2_counts = df['stage2_label'].value_counts().sort_values(ascending=False)
for val, count in stage2_counts.items():
    percentage = count / len(df) * 100
    bar = "█" * int(percentage / 2)
    print(f"  {val:25s} {count:4d} ({percentage:5.1f}%) {bar}")

# Show validation set distribution
print(f"\n✅ VALIDATION SET (What Models are Tested On)")
print("=" * 80)
val_df['stage1_label'] = val_df['Main Type'].apply(get_stage1_label)
val_df['stage2_label'] = val_df['Main Type'].apply(get_stage2_label)

print(f"Stage 1 Validation:")
val_stage1_counts = val_df['stage1_label'].value_counts().sort_values(ascending=False)
for val, count in val_stage1_counts.items():
    percentage = count / len(val_df) * 100
    bar = "█" * int(percentage / 2)
    print(f"  {val:25s} {count:4d} ({percentage:5.1f}%) {bar}")

# Key metrics
print(f"\n📊 CLASS BALANCE ANALYSIS")
print("=" * 80)
benign_count = len(df[df['stage1_label'] == 'Benign'])
harmful_count = len(df[df['stage1_label'] == 'Harmful Content'])
attack_count = len(df[df['stage1_label'] == 'Attack'])

print(f"  Stage 1 (Intent):")
print(f"    • Benign: {benign_count} ({benign_count/len(df)*100:.1f}%)")
print(f"    • Harmful Content: {harmful_count} ({harmful_count/len(df)*100:.1f}%)")
print(f"    • Attack: {attack_count} ({attack_count/len(df)*100:.1f}%)")

semantic_count = len(df[df['stage2_label'] == 'Semantic Attack'])
obfuscation_count = len(df[df['stage2_label'] == 'Obfuscation Attack'])
vector_count = len(df[df['stage2_label'] == 'Vector Attack'])
print(f"\n  Stage 2 (Attack Types):")
print(f"    • Semantic: {semantic_count} ({semantic_count/attack_count*100:.1f}% of attacks)")
print(f"    • Obfuscation: {obfuscation_count} ({obfuscation_count/attack_count*100:.1f}% of attacks)")
print(f"    • Vector: {vector_count} ({vector_count/attack_count*100:.1f}% of attacks)")

# Identify imbalance issues
print(f"\n⚠️  CLASS IMBALANCE ANALYSIS")
print("=" * 80)
ideal_stage1 = len(df) / 3  # Each stage 1 class should be ~33%
ideal_attack = attack_count / 3  # Each attack type should be ~33% of attacks

if harmful_count < ideal_stage1 * 0.6:
    deficit = int(ideal_stage1 - harmful_count)
    print(f"  ⚠️  Harmful Content underrepresented")
    print(f"     Current: {harmful_count} ({harmful_count/len(df)*100:.1f}%)")
    print(f"     Ideal: ~{int(ideal_stage1)} (33.3%)")
    print(f"     → Need {deficit} more Harmful Content samples")

if vector_count < ideal_attack * 0.5:
    deficit = int(ideal_attack - vector_count)
    print(f"\n  ❌ Vector attacks severely underrepresented")
    print(f"     Current: {vector_count} ({vector_count/attack_count*100:.1f}%)")
    print(f"     Ideal: ~{int(ideal_attack)} (33.3%)")
    print(f"     → Need {deficit} more Vector attack samples")

if semantic_count > obfuscation_count * 1.5:
    print(f"\n  ⚠️  Semantic attacks dominate Obfuscation")
    print(f"     Ratio: {semantic_count/obfuscation_count:.1f}:1")
    print(f"     → Model may prefer Semantic classification")

if semantic_count > vector_count * 3:
    print(f"\n  ⚠️  Semantic attacks dominate Vector")
    print(f"     Ratio: {semantic_count/vector_count:.1f}:1")
    print(f"     → Model may misclassify Vector as Semantic")

print("\n" + "=" * 80)
