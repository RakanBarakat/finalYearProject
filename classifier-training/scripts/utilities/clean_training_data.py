"""
Script to clean mislabeled samples in final_augmented_dataset.csv
Identifies Benign samples that contain harmful keywords and need review
"""

import pandas as pd

df = pd.read_csv("final_augmented_dataset.csv")

print(f"Total samples: {len(df)}")
print(f"Benign samples: {len(df[df['Main Type'] == 'Benign'])}")

# Keywords that warrant review (not all indicate mislabeling - many are idioms/gaming)
suspicious_keywords = {
    'kill': 'often idiom (kill for coffee) or gaming (kill zombies)',
    'assassinate': 'gaming context typically',
    'bomb': 'often idiom (bombed the test) or gaming',
    'hack': 'could be asking about security OR gaming (Fallout, etc)',
    'steal': 'could be gaming or general inquiry',
    'fraud': 'could be asking to commit OR report',
    'scam': 'could be asking about OR how to identify',
    'exploit': 'could be security vulnerability OR gaming',
    'vulnerabilit': 'security context - needs review',
    'attack': 'could be cyber OR gaming OR current events',
}

print("\n=== SAMPLES REQUIRING MANUAL REVIEW ===\n")

samples_to_review = []

for idx, row in df.iterrows():
    if row['Main Type'] != 'Benign':
        continue
    
    text = str(row['text']).lower()
    
    for keyword, note in suspicious_keywords.items():
        if keyword in text:
            samples_to_review.append({
                'idx': idx,
                'text': row['text'],
                'subtype': row['Sub-Type'],
                'matched_keyword': keyword,
                'note': note
            })
            break

print(f"Samples requiring review: {len(samples_to_review)}\n")

# Display samples for manual review
for i, sample in enumerate(samples_to_review[:30], 1):
    print(f"{i}. [Idx: {sample['idx']}] [Sub-Type: {sample['subtype']}]")
    print(f"   Keyword: '{sample['matched_keyword']}' - {sample['note']}")
    print(f"   Text: {sample['text'][:100]}...")
    print()

# Create recommendations
print("\n=== AUTOMATIC RECOMMENDATIONS ===\n")

# Samples that are clearly benign (idioms, gaming, general questions)
clearly_benign = []
for s in samples_to_review:
    text = s['text'].lower()
    
    # Idioms
    if any(phrase in text for phrase in [
        'kill for',  # "kill for coffee/sleep"
        'killing it',  # idiom for doing well
        'killed me',  # idiom (funny)
        'bombed that test',  # idiom
    ]):
        clearly_benign.append((s['idx'], s['text'], "Benign - idiom"))
    
    # Gaming context
    elif any(word in text for word in [
        'fortnite', 'resident evil', 'zombie', 'skyrim', 
        'fallout', 'hitman', 'game'
    ]):
        if any(kw in text for kw in ['how to', 'tips for', 'strategy']):
            clearly_benign.append((s['idx'], s['text'], "Gaming context - likely Benign"))

# Samples that need expert review
needs_review = []
for s in samples_to_review:
    is_clearly_benign = False
    for cb in clearly_benign:
        if cb[0] == s['idx']:
            is_clearly_benign = True
            break
    
    if not is_clearly_benign:
        needs_review.append((s['idx'], s['text'], s['subtype']))

print(f"Clearly Benign (keep as is): {len(clearly_benign)}")
for idx, text, reason in clearly_benign[:10]:
    print(f"  [{idx}] {text[:60]}... ({reason})")

print(f"\nNeeds Expert Review: {len(needs_review)}")
for idx, text, subtype in needs_review[:15]:
    print(f"  [{idx}] [{subtype}] {text[:60]}...")

# Export samples needing review for manual decision
review_df = pd.DataFrame([{
    'idx': s['idx'],
    'text': s['text'],
    'subtype': s['subtype'],
    'matched_keyword': s['matched_keyword'],
    'recommended_label': '',  # To be filled manually
    'notes': ''
} for s in samples_to_review])

review_df.to_csv("samples_requiring_review.csv", index=False)
print(f"\nExported {len(review_df)} samples to 'samples_requiring_review.csv' for manual review")

# Option: Auto-fix clearly benign samples (uncomment to apply)
print("\n=== APPLYING AUTO-FIXES ===")
print("Uncomment the lines below to apply automatic fixes:\n")

print("""
# Apply fixes for clearly benign samples
fix_indices = [idx for idx, _, _ in clearly_benign]
df.loc[fix_indices, 'Main Type'] = 'Benign'
df.loc[fix_indices, 'Sub-Type'] = 'Benign'

# For samples needing review, manually decide based on context
# Then apply changes:
# df.loc[new_idx, 'Main Type'] = 'Harmful Content Requests'  # If truly harmful
# df.loc[new_idx, 'Main Type'] = 'Benign'  # If truly benign

# Save cleaned dataset
df.to_csv("final_augmented_dataset_cleaned.csv", index=False)
""")
