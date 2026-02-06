import pandas as pd

df = pd.read_csv('combined_prompt_injection_dataset.csv')

print('=== DATASET OVERVIEW ===')
print(f'Total rows: {len(df)}')
print(f'Total columns: {len(df.columns)}')
print(f'\nColumns: {df.columns.tolist()}')

print('\n=== NULL VALUES ===')
print(df.isnull().sum())

print('\n=== MAIN TYPE DISTRIBUTION ===')
print(df['Main Type'].value_counts())

print('\n=== SUB-TYPE DISTRIBUTION ===')
print(df['Sub-Type'].value_counts())

print('\n=== SAMPLE ROWS ===')
sample_df = df.sample(5)[['text', 'Main Type', 'Sub-Type']].copy()
sample_df['text'] = sample_df['text'].str[:60] + '...'
print(sample_df.to_string())

print('\n=== DATA QUALITY CHECKS ===')
print(f'Empty text fields: {(df["text"].astype(str).str.strip() == "").sum()}')
print(f'Duplicate texts: {df["text"].duplicated().sum()}')
print(f'Rows with "0" in Main Type: {(df["Main Type"] == "0").sum()}')
print(f'Rows with "0" in Sub-Type: {(df["Sub-Type"] == "0").sum()}')
question_pattern = r"\?"
print(f'Main Type with "?" character: {df["Main Type"].astype(str).str.contains(question_pattern, na=False).sum()}')
print(f'Sub-Type with "?" character: {df["Sub-Type"].astype(str).str.contains(question_pattern, na=False).sum()}')

print('\n=== UNIQUE VALUES ===')
print(f'Unique Main Types: {df["Main Type"].nunique()}')
print(f'Unique Sub-Types: {df["Sub-Type"].nunique()}')

print('\n✓ Dataset validation complete!')
