# Evaluation Dataset Balance

## Overview

| Dataset | Type | Approx. Size | Purpose |
|---------|------|--------------|---------|
| **notInject** | Benign only | ~339 | Test false positive rate (FPR) — guardrail should allow legitimate prompts |
| **StrongREJECT** | Malicious only | ~300+ | Jailbreak/prompt-injection attacks |
| **vectorAttacks** | Mixed | ~656 (601 mal + 55 ben) | Obfuscation, vector (spaced chars), other evasion |
| **ProtectAI** | Mixed | ~3,227 | Large validation set: benign + injection |

## Balance Assessment

**Yes, this is a balanced evaluation set** for guardrail testing:

1. **Benign coverage**: notInject (339) + ProtectAI benign + vectorAttacks benign — tests FPR
2. **Malicious coverage**: StrongREJECT (jailbreaks) + vectorAttacks (obfuscation/vector) + ProtectAI injection — tests recall across attack types
3. **Attack diversity**: 
   - **Obfuscation**: Unicode, leetspeak (vectorAttacks)
   - **Vector**: Spaced-character attacks (vectorAttacks)
   - **Prompt injection**: Jailbreaks, role-play (StrongREJECT, ProtectAI)

## Combined Metrics

When running "Compare All", metrics are aggregated by summing TP, TN, FP, FN across all datasets, then computing accuracy, precision, recall, F1, FPR. This gives an overall performance score across the full evaluation suite.
