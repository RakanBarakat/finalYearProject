================================================================================
TASK 3 & 4: 2-CLASS MODEL WITH "UNSAFE" LABEL - IMPLEMENTED
================================================================================

CREATED FILES:
1. final_augmented_dataset_2class.csv - 2-class labeled dataset
2. finetune_promptguard_2class.py - Training script for 2-class model
3. final_augmented_dataset_2class_mapping.json - Label mapping reference

--------------------------------------------------------------------------------
DATASET STATS:
--------------------------------------------------------------------------------
Total samples: 5,157
  - Benign: 2,775 (53.8%)
  - Unsafe: 2,382 (46.2%)

LABEL MAPPING:
  0 = Benign   (safe, non-harmful prompts)
  1 = Unsafe   (all attacks, injections, harmful content)

ORIGINAL → NEW MAPPING:
  Benign                           → Benign
  Harmful Content Requests         → Unsafe
  Semantic & Logic Attacks          → Unsafe
  Evasion & Obfuscation Attacks    → Unsafe
  Advanced Vector Attacks          → Unsafe
  Prompt Injection                 → Unsafe

--------------------------------------------------------------------------------
WHY "UNSAFE" IS BETTER THAN "ATTACK":
--------------------------------------------------------------------------------
- "Attack" implies a security breach or injection attempt
- "Unsafe" encompasses both:
  * Direct harmful content (malware creation, violence instructions)
  * Indirect attacks (role-play framing, hypotheticals)
  * Injection attempts (ignoring system prompts)
- More intuitive: a prompt is either Safe/Unsafe rather than Attack/Non-Attack
- Matches common security terminology (Safe/Unsafe, Low/High Risk)

--------------------------------------------------------------------------------
TRAINING:
--------------------------------------------------------------------------------
Run:
  cd classifier-training
  python finetune_promptguard_2class.py

This creates: ./custom_prompt_guard_2class/ (checkpoints)
Final model:  ./final_security_model_2class/

--------------------------------------------------------------------------------
UPDATING YOUR GUARDRAILS APP:
--------------------------------------------------------------------------------
After training, zip the model folder and upload to your guardrails app.

The model's config.json will have:
  "id2label": {"0": "Benign", "1": "Unsafe"}
  "num_labels": 2

Your existing main.py already handles 2-class models correctly via
risk_score_from_probs() which calculates non-benign probability.

--------------------------------------------------------------------------------
EXPECTED IMPROVEMENTS:
--------------------------------------------------------------------------------
1. Simpler decision boundary (2 classes vs 5 attack types)
2. Better generalization to unseen attacks
3. Improved recall on StrongREJECT (direct harmful content → Unsafe)
4. Reduced false negatives on sophisticated jailbreaks

The classifier now only needs to learn:
  - "Is this prompt Benign?" → YES (class 0)
  - "Is this prompt Benign?" → NO (class 1 = Unsafe)

Much simpler than learning to distinguish 5 different attack categories!
================================================================================
