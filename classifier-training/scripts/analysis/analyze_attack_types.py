import pandas as pd

df = pd.read_csv('original_attacks_dataset.csv')

print('=' * 80)
print('DETAILED ATTACK TYPE ANALYSIS')
print('=' * 80)

# Analyze the 3 main categories
print('\n1. BENIGN SAMPLES')
print('-' * 80)
benign = df[df['Main Type'] == 'Benign']
print(f'Total: {len(benign)} samples ({len(benign)/len(df)*100:.1f}%)')
print(f'\nSub-types:')
print(benign['Sub-Type'].value_counts())
print(f'\nExample benign texts:')
for i, text in enumerate(benign['text'].sample(10, random_state=42).tolist(), 1):
    display = text[:120] + '...' if len(text) > 120 else text
    print(f'  {i}. {display}')

print('\n\n2. EVASION & OBFUSCATION ATTACKS')
print('-' * 80)
evasion = df[df['Main Type'] == 'Evasion & Obfuscation Attacks']
print(f'Total: {len(evasion)} samples ({len(evasion)/len(df)*100:.1f}%)')
print(f'\nSub-types breakdown:')
sub_counts = evasion['Sub-Type'].value_counts()
for sub_type, count in sub_counts.items():
    percentage = (count / len(evasion)) * 100
    print(f'  - {sub_type}: {count} ({percentage:.1f}%)')

print(f'\nExample attacks by sub-type:')
for sub_type in sub_counts.index[:4]:  # Show top 4 sub-types
    print(f'\n  [{sub_type}]')
    samples = evasion[evasion['Sub-Type'] == sub_type]['text'].head(2).tolist()
    for text in samples:
        display = text[:100] + '...' if len(text) > 100 else text
        print(f'    • {display}')

print('\n\n3. SEMANTIC & LOGIC ATTACKS')
print('-' * 80)
semantic = df[df['Main Type'] == 'Semantic & Logic Attacks']
print(f'Total: {len(semantic)} samples ({len(semantic)/len(df)*100:.1f}%)')
print(f'\nSub-types breakdown:')
sub_counts = semantic['Sub-Type'].value_counts()
for sub_type, count in sub_counts.items():
    percentage = (count / len(semantic)) * 100
    print(f'  - {sub_type}: {count} ({percentage:.1f}%)')

print(f'\nExample attacks by sub-type:')
for sub_type in sub_counts.index[:4]:  # Show top 4 sub-types
    print(f'\n  [{sub_type}]')
    samples = semantic[semantic['Sub-Type'] == sub_type]['text'].head(2).tolist()
    for text in samples:
        display = text[:100] + '...' if len(text) > 100 else text
        print(f'    • {display}')

# Analyze synthetic dataset
print('\n\n4. SYNTHETIC DATASET (Advanced Vector Attacks)')
print('-' * 80)
try:
    df_synthetic = pd.read_csv('real_and_synthetic_dataset.csv')
    advanced = df_synthetic[df_synthetic['Main Type'] == 'Advanced Vector Attacks']
    print(f'Total: {len(advanced)} samples')
    print(f'\nSub-types breakdown:')
    sub_counts = advanced['Sub-Type'].value_counts()
    for sub_type, count in sub_counts.items():
        percentage = (count / len(advanced)) * 100
        print(f'  - {sub_type}: {count} ({percentage:.1f}%)')
    
    print(f'\nExample attacks by sub-type:')
    for sub_type in sub_counts.index:
        print(f'\n  [{sub_type}]')
        samples = advanced[advanced['Sub-Type'] == sub_type]['text'].head(2).tolist()
        for text in samples:
            display = text[:100] + '...' if len(text) > 100 else text
            print(f'    • {display}')
except FileNotFoundError:
    print('Synthetic dataset not found')

# Overall statistics
print('\n\n' + '=' * 80)
print('TRAINING DATA SUMMARY')
print('=' * 80)
print(f'\nOriginal Dataset: {len(df)} samples')
print(f'  ├─ Benign: {len(benign)} ({len(benign)/len(df)*100:.1f}%)')
print(f'  ├─ Evasion & Obfuscation: {len(evasion)} ({len(evasion)/len(df)*100:.1f}%)')
print(f'  └─ Semantic & Logic: {len(semantic)} ({len(semantic)/len(df)*100:.1f}%)')

try:
    print(f'\nSynthetic Dataset: {len(df_synthetic)} samples')
    print(f'  ├─ Benign: {len(df_synthetic[df_synthetic["Main Type"] == "Benign"])}')
    print(f'  └─ Advanced Vectors (IPI/Multimodal): {len(df_synthetic[df_synthetic["Main Type"] == "Advanced Vector Attacks"])}')
    print(f'\nCombined Total: {len(df) + len(df_synthetic)} samples')
except:
    print(f'\nCombined Total: {len(df)} samples (no synthetic data)')

# Check class balance
print('\n\n' + '=' * 80)
print('CLASS BALANCE ANALYSIS')
print('=' * 80)
all_sub_types = df['Sub-Type'].value_counts()
print(f'\nTotal unique sub-types: {len(all_sub_types)}')
print(f'\nSub-type distribution:')
for sub_type, count in all_sub_types.items():
    percentage = (count / len(df)) * 100
    bar = '█' * int(percentage / 2)
    print(f'{sub_type:40s} {count:4d} ({percentage:5.1f}%) {bar}')

print('\n\nRecommendation for training:')
if len(benign) / len(df) > 0.5:
    print('✓ Good class balance - benign samples are well-represented')
else:
    print('⚠ Consider adding more benign samples to balance the dataset')

if len(all_sub_types) <= 15:
    print(f'✓ Manageable number of sub-types ({len(all_sub_types)}) for multi-class classification')
else:
    print(f'⚠ High number of sub-types ({len(all_sub_types)}) - may be challenging for model')
