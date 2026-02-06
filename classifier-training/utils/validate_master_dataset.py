import pandas as pd

def validate_dataset():
    print("=== MASTER DATASET VALIDATION ===\n")
    
    # Load dataset
    df = pd.read_csv('master_prompt_injection_dataset.csv')
    
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # Check for null values
    print("=== NULL VALUES ===")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("✓ No null values found")
    else:
        print("✗ Null values found:")
        print(null_counts[null_counts > 0])
    
    # Check for empty strings
    print("\n=== EMPTY TEXT FIELDS ===")
    empty_text = (df['text'].astype(str).str.strip() == '').sum()
    if empty_text == 0:
        print("✓ No empty text fields")
    else:
        print(f"✗ Found {empty_text} empty text fields")
    
    # Validate Main Type values
    print("\n=== MAIN TYPE VALIDATION ===")
    valid_main_types = [
        'Benign',
        'Semantic & Logic Attacks',
        'Evasion & Obfuscation Attacks',
        'Advanced Vector Attacks'
    ]
    
    actual_main_types = df['Main Type'].unique()
    print(f"Expected Main Types: {len(valid_main_types)}")
    print(f"Actual Main Types: {len(actual_main_types)}")
    
    invalid_main = [mt for mt in actual_main_types if mt not in valid_main_types]
    if invalid_main:
        print(f"✗ Invalid Main Types found: {invalid_main}")
        print(f"  Count: {df['Main Type'].isin(invalid_main).sum()} rows")
    else:
        print("✓ All Main Types are valid")
    
    print("\nMain Type Distribution:")
    print(df['Main Type'].value_counts())
    
    # Validate Sub-Type values
    print("\n=== SUB-TYPE VALIDATION ===")
    valid_sub_types = [
        'Non-Attack',
        'Hypothetical / Benign Framing',
        'Authority & Privilege Claims',
        'Role-Playing / Persona Hijacking',
        'Refusal Suppression',
        'Encoding & Representation',
        'Adversarial Suffixes (GCG)',
        'Word & Character Manipulation',
        'Indirect Prompt Injection (IPI)',
        'Multimodal Injection'
    ]
    
    actual_sub_types = df['Sub-Type'].unique()
    print(f"Expected Sub-Types: {len(valid_sub_types)}")
    print(f"Actual Sub-Types: {len(actual_sub_types)}")
    
    invalid_sub = [st for st in actual_sub_types if st not in valid_sub_types]
    if invalid_sub:
        print(f"✗ Invalid Sub-Types found: {invalid_sub}")
        print(f"  Count: {df['Sub-Type'].isin(invalid_sub).sum()} rows")
    else:
        print("✓ All Sub-Types are valid")
    
    print("\nSub-Type Distribution:")
    print(df['Sub-Type'].value_counts())
    
    # Validate Category values
    print("\n=== CATEGORY VALIDATION ===")
    valid_categories = [
        'Original_Benign',
        'Original_Semantic_Attacks',
        'Original_Obfuscation_Attacks',
        'Real_Obfuscation',
        'Synthetic_Benign',
        'Synthetic_Malicious'
    ]
    
    actual_categories = df['Category'].unique()
    print(f"Expected Categories: {len(valid_categories)}")
    print(f"Actual Categories: {len(actual_categories)}")
    
    invalid_cat = [cat for cat in actual_categories if cat not in valid_categories]
    if invalid_cat:
        print(f"✗ Invalid Categories found: {invalid_cat}")
        print(f"  Count: {df['Category'].isin(invalid_cat).sum()} rows")
    else:
        print("✓ All Categories are valid")
    
    print("\nCategory Distribution:")
    print(df['Category'].value_counts())
    
    # Check for duplicates
    print("\n=== DUPLICATE CHECK ===")
    duplicates = df['text'].duplicated().sum()
    if duplicates == 0:
        print("✓ No duplicate texts found")
    else:
        print(f"⚠ Found {duplicates} duplicate texts")
    
    # Validate logic consistency
    print("\n=== LOGIC CONSISTENCY CHECK ===")
    
    # Check: Benign Main Type should have Non-Attack Sub-Type
    benign_check = df[df['Main Type'] == 'Benign']
    benign_non_attack = (benign_check['Sub-Type'] == 'Non-Attack').sum()
    if benign_non_attack == len(benign_check):
        print("✓ All Benign samples have Non-Attack sub-type")
    else:
        print(f"✗ {len(benign_check) - benign_non_attack} Benign samples have wrong sub-type")
    
    # Check: Non-Attack Sub-Type should have Benign Main Type
    non_attack_check = df[df['Sub-Type'] == 'Non-Attack']
    non_attack_benign = (non_attack_check['Main Type'] == 'Benign').sum()
    if non_attack_benign == len(non_attack_check):
        print("✓ All Non-Attack samples have Benign main type")
    else:
        print(f"✗ {len(non_attack_check) - non_attack_benign} Non-Attack samples have wrong main type")
    
    # Check: Advanced Vector Attacks should only have IPI or Multimodal sub-types
    advanced_check = df[df['Main Type'] == 'Advanced Vector Attacks']
    valid_advanced_subs = advanced_check['Sub-Type'].isin(['Indirect Prompt Injection (IPI)', 'Multimodal Injection']).sum()
    if valid_advanced_subs == len(advanced_check):
        print("✓ All Advanced Vector Attacks have correct sub-types")
    else:
        print(f"✗ {len(advanced_check) - valid_advanced_subs} Advanced Vector Attacks have wrong sub-type")
    
    # Summary
    print("\n" + "="*60)
    print("=== VALIDATION SUMMARY ===")
    
    all_checks = [
        null_counts.sum() == 0,
        empty_text == 0,
        len(invalid_main) == 0,
        len(invalid_sub) == 0,
        len(invalid_cat) == 0,
        benign_non_attack == len(benign_check),
        non_attack_benign == len(non_attack_check),
        valid_advanced_subs == len(advanced_check)
    ]
    
    if all(all_checks):
        print("✓ ALL VALIDATION CHECKS PASSED!")
    else:
        print(f"⚠ {sum(not check for check in all_checks)} validation check(s) failed")
        print("Please review the issues above and fix the dataset.")
    
    print("="*60)

if __name__ == "__main__":
    validate_dataset()
