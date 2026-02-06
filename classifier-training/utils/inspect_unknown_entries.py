import pandas as pd

df = pd.read_csv('combined_prompt_injection_dataset.csv')

# Filter for Unknown entries
unknown_df = df[df['Main Type'] == 'Unknown'].copy()

print(f"=== FOUND {len(unknown_df)} UNKNOWN ENTRIES ===\n")

# Display each unknown entry with truncated text for readability
for idx, row in unknown_df.iterrows():
    print(f"Row {idx}:")
    print(f"Text: {row['text'][:200]}...")  # First 200 chars
    print(f"Main Type: {row['Main Type']}")
    print(f"Sub-Type: {row['Sub-Type']}")
    print("-" * 80)
    print()

# Also save full unknown entries to a separate file for easier review
unknown_df.to_csv('unknown_entries_for_review.csv', index=False)
print(f"\n✓ Full unknown entries saved to 'unknown_entries_for_review.csv'")
