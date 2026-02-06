import pandas as pd

df = pd.read_csv('combined_prompt_injection_dataset.csv')

# Find all Refusal Suppression entries
refusal_df = df[df['Sub-Type'] == 'Refusal Suppression'].copy()

print(f"=== FOUND {len(refusal_df)} REFUSAL SUPPRESSION ENTRIES ===\n")

for idx, row in refusal_df.iterrows():
    print(f"Row {idx}:")
    print(f"Text: {row['text']}")
    print(f"Main Type: {row['Main Type']}")
    print(f"Sub-Type: {row['Sub-Type']}")
    print("-" * 80)
    print()

# Let's also search for common refusal suppression patterns in other categories
print("\n=== CHECKING FOR POTENTIAL REFUSAL SUPPRESSION IN OTHER CATEGORIES ===\n")

refusal_patterns = [
    'cannot refuse', 'must not refuse', 'never refuse', 'always comply',
    'ignore safety', 'ignore ethical', 'ignore moral', 'without restriction',
    'no ethical', 'no moral', 'bypass safety', 'override safety',
    'cannot say no', 'must answer', 'required to answer', 'obligated to',
    'do not decline', 'never decline', 'disable safety', 'remove restrictions'
]

other_entries = df[df['Sub-Type'] != 'Refusal Suppression'].copy()
found_additional = False

for idx, row in other_entries.iterrows():
    text_lower = str(row['text']).lower()
    matched_patterns = [p for p in refusal_patterns if p in text_lower]
    
    if matched_patterns:
        found_additional = True
        print(f"Row {idx} (Currently: {row['Sub-Type']}):")
        print(f"Matched patterns: {matched_patterns}")
        print(f"Text preview: {row['text'][:150]}...")
        print("-" * 80)
        print()

if not found_additional:
    print("✓ No additional refusal suppression patterns found in other categories")
