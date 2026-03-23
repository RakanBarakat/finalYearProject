"""
label_eval_categories.py
=========================
Pre-fills a 'category' column in the evaluation CSVs using the same heuristic
as the backend (_classify_vector_attack_type). Benign samples (label=0) get
category='benign'. Malicious samples (label=1) get a heuristic-assigned category.

IMPORTANT: The heuristic is approximate. After running this script, manually
review the output CSVs and correct any mis-categorised rows before using
them for dissertation analysis.

Usage:
    python label_eval_categories.py

Outputs:
    - data/evaluation/prompt_injection_eval.csv  (updated in-place with 'category' column)
    - data/evaluation/vector_obfuscation_eval.csv (updated in-place with 'category' column)

Categories:
    - benign: Ground-truth benign (label=0)
    - prompt_injection: Direct injection / jailbreak keywords
    - obfuscation: Unicode tricks, encoding (base64/ROT13/hex), leetspeak, mixed case
    - vector: Document/image context-wrapped attacks
    - harmful: Unclassified harmful/malicious samples
"""

import csv
import os
import re
import sys


def classify_attack_type(text: str) -> str:
    """
    Heuristic classification of malicious prompts into attack sub-categories.
    Mirrors the backend _classify_vector_attack_type() function.
    """
    if not text or not isinstance(text, str):
        return "harmful"
    t = text[:2000].lower()
    raw = text[:2000]

    # Vector: document/image context-wrapped attacks
    vector_phrases = (
        "[source:", "[ocr]", "document text:", "document:", "pdf content", "pdf:",
        "image/ocr", "[image/", "[audio/", "meeting transcript", "transcript:",
        "email i received", "here is the email", "attachment:", "from: admin",
        "body:", "[extracted text]", "[log output]", "source: text/plain",
        "source: message/", "source: image/", "ocr output:", "translated text from",
        "read the following file", "analyze this", "summarize this document",
        "scanned this document", "image says", "transcribed the audio",
        "chat message from", "support ticket", "form submission",
        "database record", "presentation slide", "parsed resume",
        "customer review", "markdown document", "sms message",
    )
    if any(p in t for p in vector_phrases):
        return "vector"

    # Obfuscation: encoding (base64/ROT13/hex), unicode tricks, leetspeak, spacing
    if "decode and execute:" in t or "rot13 decode:" in t or "rot13 decode " in t or "base64" in t:
        return "obfuscation"
    if re.search(r"[A-Za-z0-9+/]{20,}={0,2}\s*$", t) or re.search(r"\b[0-9a-fA-F]{16,}\b", t):
        if "decode" in t or "execute" in t or "follow:" in t:
            return "obfuscation"

    has_zero_width = "\u200b" in raw or "\u200c" in raw or "\u200d" in raw or "\ufeff" in raw
    _UPSIDE_DOWN = set("ɐɔǝɟƃɥᴉɾʞɯɹʇnʌʍʎ˙¿¡")
    upside_down_chars = sum(1 for c in raw if c in _UPSIDE_DOWN)
    non_ascii_count = sum(1 for c in raw if ord(c) > 127)
    non_ascii_ratio = non_ascii_count / max(len(raw), 1)
    extra_spacing = bool(re.search(r"\w\s{2,}\w\s{2,}\w", raw))
    leetspeak_matches = re.findall(r"[a-zA-Z][0-9][a-zA-Z]|[0-9][a-zA-Z][0-9]", raw)
    leetspeak = len(leetspeak_matches) >= 3
    mixed_case_ratio = 0
    alpha_chars = [c for c in raw if c.isalpha()]
    if len(alpha_chars) > 10:
        case_switches = sum(1 for a, b in zip(alpha_chars, alpha_chars[1:]) if a.isupper() != b.isupper())
        mixed_case_ratio = case_switches / len(alpha_chars)

    if has_zero_width or upside_down_chars > 5:
        return "obfuscation"
    if non_ascii_ratio > 0.15 and non_ascii_count > 10:
        return "obfuscation"
    if leetspeak or extra_spacing:
        return "obfuscation"
    if mixed_case_ratio > 0.4:
        return "obfuscation"

    # Prompt injection: direct jailbreak / instruction override keywords
    injection_phrases = (
        "ignore previous", "ignore all", "disregard", "bypass", "override", "system prompt",
        "you are now", "assume persona", "act as dan", "do anything now", "developer mode",
        "for the rest of this conversation", "supersede", "new instructions",
        "ignore your", "forget your", "new role", "pretend you", "roleplay as",
        "jailbreak", "unrestricted", "no restrictions", "no censorship", "no filter",
        "skip all safety", "skip safety", "mandatory task", "compliance is mandatory",
        "emergency protocol", "override safety", "new directive",
    )
    if any(p in t for p in injection_phrases):
        return "prompt_injection"

    return "harmful"


def label_csv(filepath: str, dataset_type: str) -> dict:
    """
    Read a CSV, add/overwrite 'category' column, write back.
    Returns category distribution stats.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    # Add category column
    if "category" not in fieldnames:
        fieldnames.append("category")

    stats = {}
    for row in rows:
        label = row.get("label", "")
        is_benign = label in ("0", "benign", "Benign")

        if is_benign:
            category = "benign"
        else:
            category = classify_attack_type(row.get("text", ""))
            # For injection-focused datasets, remap unclassified to prompt_injection
            if dataset_type == "injection" and category == "harmful":
                category = "prompt_injection"

        row["category"] = category
        stats[category] = stats.get(category, 0) + 1

    # Write back
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return stats


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, "..", "..", "data", "evaluation")

    files = [
        ("prompt_injection_eval.csv", "injection"),
        ("vector_obfuscation_eval.csv", "vector"),
    ]

    for filename, dtype in files:
        filepath = os.path.join(datasets_dir, filename)
        if not os.path.exists(filepath):
            print(f"[SKIP] {filename} not found at {filepath}")
            continue

        print(f"\n{'='*60}")
        print(f"Labeling: {filename} (type={dtype})")
        print(f"{'='*60}")

        stats = label_csv(filepath, dtype)

        total = sum(stats.values())
        print(f"Total rows: {total}")
        print(f"\nCategory distribution:")
        for cat in ["benign", "prompt_injection", "obfuscation", "vector", "harmful"]:
            count = stats.get(cat, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n{'='*60}")
    print("DONE. Categories pre-filled using heuristic.")
    print("IMPORTANT: Review the CSVs and correct any mis-categorised rows.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
