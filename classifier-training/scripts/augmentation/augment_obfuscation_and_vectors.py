"""
Balance the training dataset by augmenting underrepresented attack categories.

Targets:
  - Evasion & Obfuscation Attacks: 375 → ~800
  - Advanced Vector Attacks:       195 → ~800

Methodology:
  1. Source payloads are sampled from EXISTING malicious prompts in the dataset
     (Harmful Content, Semantic & Logic Attacks, Prompt Injection). No hand-crafted
     prompts are introduced — only transformations of real training data.
  2. Each augmented sample receives a RANDOMLY selected technique (seeded for
     reproducibility). No sample is given a fixed technique based on position.
  3. Encoded content (Base64, ROT13) is stored RAW — no wrapper prefixes like
     "Decode and execute:". The classifier must learn the encoding character
     patterns, not a human-readable label.
  4. Vector attack templates are drawn from a pool of 25+ document contexts with
     randomised payload placement (start, middle, end) and varied filler text.
  5. All new samples are deduplicated against existing training data using
     normalised fingerprinting.

No external data sources are used. All augmented samples are transformations
of prompts already present in the training set.

Usage:
    cd classifier-training
    python scripts/augmentation/augment_obfuscation_and_vectors.py
"""

import base64
import codecs
import csv
import os
import random
import re
import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
TARGET_OBFUSCATION = 800
TARGET_VECTOR = 800
TARGET_BENIGN_VECTOR = 400

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "training")

INPUT_CSV = os.path.join(DATA_DIR, "final_augmented_dataset_cleaned.csv")
OUTPUT_CSV = INPUT_CSV  # overwrite in place

SOURCE_CATEGORIES = [
    "Harmful Content Requests",
    "Semantic & Logic Attacks",
    "Prompt Injection",
    "Evasion & Obfuscation Attacks",
    "Advanced Vector Attacks",
]

# ---------------------------------------------------------------------------
# Utility (mirrors build_rag_index.py for consistency)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def fingerprint(text: str) -> str:
    t = normalize_text(text).lower()
    t = re.sub(r"[\u200b-\u200f\ufeff]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_high_quality(text: str) -> bool:
    t = normalize_text(text)
    if len(t) < 12 or len(t) > 2500:
        return False
    if t.lower() in {"text", "main type", "sub-type", "specific type", "category"}:
        return False
    if t.count("\ufffd") > 0:
        return False
    alpha = sum(ch.isalpha() for ch in t)
    if alpha < 4:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# OBFUSCATION TECHNIQUES  (8 techniques, no wrapper prefixes)
# ═══════════════════════════════════════════════════════════════════════════

def apply_base64(text: str, rng: random.Random) -> str:
    """Raw Base64 encoding — no prefix, no instruction, just the encoded string."""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def apply_rot13(text: str, rng: random.Random) -> str:
    """Raw ROT13 — cipher-text only, no label."""
    return codecs.encode(text, "rot_13")


def apply_leetspeak(text: str, rng: random.Random) -> str:
    """Replace a random 50-90% of eligible characters with leet equivalents."""
    leet_map = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
    rate = rng.uniform(0.5, 0.9)
    out = []
    for ch in text:
        if ch.lower() in leet_map and rng.random() < rate:
            out.append(leet_map[ch.lower()])
        else:
            out.append(ch)
    return "".join(out)


def apply_homoglyph(text: str, rng: random.Random) -> str:
    """Replace 30-70% of eligible Latin characters with Cyrillic lookalikes."""
    glyph_map = {
        "a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440",
        "c": "\u0441", "x": "\u0445", "y": "\u0443", "i": "\u0456",
        "A": "\u0410", "E": "\u0415", "O": "\u041e", "P": "\u0420",
        "C": "\u0421", "X": "\u0425",
    }
    rate = rng.uniform(0.3, 0.7)
    out = []
    for ch in text:
        if ch in glyph_map and rng.random() < rate:
            out.append(glyph_map[ch])
        else:
            out.append(ch)
    return "".join(out)


def apply_zero_width(text: str, rng: random.Random) -> str:
    """Insert 4-10 zero-width characters at random positions within the text."""
    zwchars = ["\u200b", "\u200c", "\u200d"]
    chars = list(text)
    n_inserts = rng.randint(4, min(10, max(4, len(chars) // 5)))
    for _ in range(n_inserts):
        pos = rng.randint(1, max(1, len(chars) - 1))
        chars.insert(pos, rng.choice(zwchars))
    return "".join(chars)


def apply_accented(text: str, rng: random.Random) -> str:
    """Replace 40-80% of vowels with randomly chosen accented variants."""
    accent_map = {
        "a": "àáâãäå", "e": "èéêë", "i": "ìíîï",
        "o": "òóôõö", "u": "ùúûü", "y": "ýÿ",
    }
    rate = rng.uniform(0.4, 0.8)
    out = []
    for ch in text:
        low = ch.lower()
        if low in accent_map and rng.random() < rate:
            replacement = rng.choice(accent_map[low])
            out.append(replacement if ch.islower() else replacement.upper())
        else:
            out.append(ch)
    return "".join(out)


def apply_fullwidth(text: str, rng: random.Random) -> str:
    """Convert ASCII characters to their fullwidth Unicode equivalents."""
    out = []
    for ch in text:
        cp = ord(ch)
        # ASCII printable range 0x21-0x7E maps to fullwidth 0xFF01-0xFF5E
        if 0x21 <= cp <= 0x7E:
            out.append(chr(cp - 0x21 + 0xFF01))
        elif ch == " ":
            out.append("\u3000")  # fullwidth space
        else:
            out.append(ch)
    return "".join(out)


def apply_case_alternation(text: str, rng: random.Random) -> str:
    """Alternate upper/lower case and optionally inject symbol separators."""
    inject_symbols = rng.choice([True, False])
    symbols = [".", "*", "!", "@", "#", "-", "_"]
    out = []
    upper = rng.choice([True, False])  # randomise starting case
    for ch in text:
        if ch.isalpha():
            out.append(ch.upper() if upper else ch.lower())
            upper = not upper
        elif ch == " " and inject_symbols and rng.random() < 0.3:
            out.append(rng.choice(symbols))
        else:
            out.append(ch)
    return "".join(out)


OBFUSCATION_TECHNIQUES = [
    (apply_base64, "Encoding & Representation"),
    (apply_rot13, "Encoding & Representation"),
    (apply_leetspeak, "Word & Character Manipulation"),
    (apply_homoglyph, "Character Manipulation & Encoding"),
    (apply_zero_width, "Character Manipulation & Encoding"),
    (apply_accented, "Character Manipulation & Encoding"),
    (apply_fullwidth, "Character Manipulation & Encoding"),
    (apply_case_alternation, "Word & Character Manipulation"),
]


# ═══════════════════════════════════════════════════════════════════════════
# VECTOR ATTACK TEMPLATES  (25 document contexts)
# ═══════════════════════════════════════════════════════════════════════════

def _place_payload(before: str, payload: str, after: str, rng: random.Random) -> str:
    """Randomly place the payload at start, middle, or end of the document."""
    pos = rng.choice(["start", "middle", "end"])
    if pos == "start":
        return f"{payload}\n\n{before}\n{after}"
    elif pos == "middle":
        return f"{before}\n\n{payload}\n\n{after}"
    else:
        return f"{before}\n{after}\n\n{payload}"


def _build_vector_templates():
    """Return a list of (template_fn, description) tuples."""
    templates = []

    # 1. Email (RFC822)
    def email_rfc(payload, rng):
        senders = ["admin@corp.com", "hr@company.org", "support@service.io", "noreply@bank.com", "security@internal.net"]
        subjects = ["Urgent", "Action Required", "Q3 Report", "Invoice #2024-001", "Policy Update", "Meeting Follow-up"]
        fillers = [
            "Please review the attached document and take appropriate action.",
            "As discussed in our meeting, please see the details below.",
            "This is an automated notification from the system.",
            "Kindly confirm receipt of this message at your earliest convenience.",
        ]
        return f"[Source: message/rfc822]\nFrom: {rng.choice(senders)}\nSubject: {rng.choice(subjects)}\n\n" + \
               _place_payload(rng.choice(fillers), payload, rng.choice(fillers), rng)
    templates.append(email_rfc)

    # 2. PDF extraction
    def pdf_extract(payload, rng):
        headers = ["[Extracted Text from PDF]", "[Source: application/pdf]\nPDF Content:", "Document extracted from uploaded PDF:"]
        fillers = [
            "Page 1 of 3. Document ID: DOC-2024-0891.",
            "Classification: Internal Use Only.",
            "Generated by Adobe Acrobat Pro DC.",
        ]
        return f"{rng.choice(headers)}\n" + _place_payload(rng.choice(fillers), payload, "", rng)
    templates.append(pdf_extract)

    # 3. OCR output (image)
    def ocr_image(payload, rng):
        sources = ["[Source: image/ocr]", "[Source: image/png]", "[Source: image/jpeg]"]
        headers = ["[OCR Output]:", "[Recognised Text]:", "[Extracted Text]:"]
        fillers = [
            "Confidence: 94.2%. Language: English.",
            "Scanned at 300 DPI. Processing time: 1.2s.",
            "Image dimensions: 1920x1080. Text regions: 3.",
        ]
        return f"{rng.choice(sources)}\n{rng.choice(headers)}\n{rng.choice(fillers)}\n\n{payload}"
    templates.append(ocr_image)

    # 4. Audio transcription
    def audio_transcript(payload, rng):
        speakers = ["[Speaker 1]", "[Agent]", "[Caller]", "[Participant A]"]
        fillers = [
            "[Speaker 2]: I understand, let me check on that.",
            "[Speaker 2]: Sure, let me look into it.",
            "[Moderator]: Moving on to the next item.",
        ]
        return f"[Source: audio/transcription]\nTranscript:\n\n{rng.choice(speakers)}: {payload}\n{rng.choice(fillers)}"
    templates.append(audio_transcript)

    # 5. Meeting transcript
    def meeting_transcript(payload, rng):
        contexts = [
            "Weekly standup meeting — 2024-03-15",
            "Board meeting minutes — Q1 Review",
            "Team retrospective — Sprint 24",
        ]
        return f"Meeting Transcript:\n{rng.choice(contexts)}\n\n" + _place_payload(
            "Attendees: John, Sarah, Mike", payload, "Action items to follow.", rng)
    templates.append(meeting_transcript)

    # 6. Log file output
    def log_output(payload, rng):
        paths = ["/var/log/syslog", "/var/log/auth.log", "C:\\Windows\\System32\\LogFiles\\app.log"]
        timestamps = ["2024-03-15 14:22:01", "2024-11-08 09:15:33", "2025-01-20 23:01:17"]
        return f"[Source: text/plain]\nLog file: {rng.choice(paths)}\n[{rng.choice(timestamps)}] {payload}"
    templates.append(log_output)

    # 7. Plain document
    def plain_document(payload, rng):
        headers = ["Document text:", "Forwarded content:", "Copied text from shared document:"]
        return f"{rng.choice(headers)}\n\n{payload}"
    templates.append(plain_document)

    # 8. Webpage content
    def webpage_content(payload, rng):
        urls = ["https://internal.wiki/page/4521", "https://docs.corp.com/policies", "https://kb.support.io/article/789"]
        return f"[Source: text/html]\nCrawled from: {rng.choice(urls)}\n\n{payload}"
    templates.append(webpage_content)

    # 9. Code review comment
    def code_review(payload, rng):
        files = ["src/auth.py", "api/routes.js", "lib/utils.go", "controllers/main.rs"]
        return f"Code review comment on {rng.choice(files)}:\n\n// {payload}"
    templates.append(code_review)

    # 10. Support ticket
    def support_ticket(payload, rng):
        ids = ["TKT-4521", "CASE-00891", "INC-2024-156", "SR-7890"]
        priorities = ["High", "Critical", "Medium", "Urgent"]
        return f"Support Ticket {rng.choice(ids)} — Priority: {rng.choice(priorities)}\n\nCustomer message:\n{payload}"
    templates.append(support_ticket)

    # 11. Form submission
    def form_submission(payload, rng):
        fields = ["feedback", "comments", "description", "additional_notes", "message"]
        return f"[Source: application/x-www-form-urlencoded]\nForm field '{rng.choice(fields)}':\n{payload}"
    templates.append(form_submission)

    # 12. Database record
    def database_record(payload, rng):
        tables = ["user_comments", "feedback_entries", "support_messages", "content_submissions"]
        return f"[Source: application/json]\nDatabase record from `{rng.choice(tables)}`:\n{{\n  \"id\": {rng.randint(1000, 9999)},\n  \"content\": \"{payload}\"\n}}"
    templates.append(database_record)

    # 13. Presentation slide
    def presentation_slide(payload, rng):
        titles = ["Strategic Overview", "Q4 Planning", "Product Roadmap", "Security Audit"]
        return f"[Extracted Text from PDF]\nSlide {rng.randint(3, 18)} — {rng.choice(titles)}\n\n{payload}"
    templates.append(presentation_slide)

    # 14. Parsed resume
    def parsed_resume(payload, rng):
        return f"[Source: application/pdf]\nParsed resume content:\n\nSummary:\n{payload}\n\nSkills: Python, JavaScript, SQL"
    templates.append(parsed_resume)

    # 15. Customer review
    def customer_review(payload, rng):
        products = ["Cloud Platform Pro", "SecureAuth v3", "DataSync Enterprise", "AI Assistant"]
        stars = rng.randint(1, 5)
        return f"Product review for {rng.choice(products)} ({'★' * stars}{'☆' * (5-stars)}):\n\n{payload}"
    templates.append(customer_review)

    # 16. Markdown document
    def markdown_doc(payload, rng):
        titles = ["# Internal Policy Update", "# Release Notes v2.4", "# Migration Guide", "# Incident Report"]
        return f"[Source: text/markdown]\n{rng.choice(titles)}\n\n{payload}\n\n---\n*Last updated: 2024-03-15*"
    templates.append(markdown_doc)

    # 17. Chat message
    def chat_message(payload, rng):
        platforms = ["Slack", "Teams", "Discord", "Internal Chat"]
        channels = ["#general", "#support", "#engineering", "#security-alerts"]
        users = ["alice.johnson", "bob.smith", "system_bot", "admin"]
        return f"[{rng.choice(platforms)} — {rng.choice(channels)}]\n{rng.choice(users)}: {payload}"
    templates.append(chat_message)

    # 18. Scanned document
    def scanned_doc(payload, rng):
        return f"[Source: image/png]\n[OCR Output]:\nScanned document — page {rng.randint(1, 8)}\n\n{payload}"
    templates.append(scanned_doc)

    # 19. XML feed
    def xml_feed(payload, rng):
        tags = ["entry", "item", "content", "description"]
        tag = rng.choice(tags)
        return f"[Source: application/xml]\n<{tag}>\n  {payload}\n</{tag}>"
    templates.append(xml_feed)

    # 20. DOCX extracted text
    def docx_extract(payload, rng):
        filenames = ["report_final.docx", "meeting_notes.docx", "policy_v3.docx", "proposal_draft.docx"]
        return f"[Source: application/vnd.openxmlformats-officedocument.wordprocessingml.document]\nExtracted from: {rng.choice(filenames)}\n\n{payload}"
    templates.append(docx_extract)

    # 21. Spreadsheet cell
    def spreadsheet_cell(payload, rng):
        cells = ["B14", "D7", "A1", "C23", "E9"]
        sheets = ["Sheet1", "Data", "Comments", "Feedback"]
        return f"[Source: text/csv]\nSpreadsheet '{rng.choice(sheets)}', cell {rng.choice(cells)}:\n{payload}"
    templates.append(spreadsheet_cell)

    # 22. API response
    def api_response(payload, rng):
        endpoints = ["/api/v2/comments", "/api/feedback", "/api/content/raw", "/api/messages"]
        return f"[Source: application/json]\nGET {rng.choice(endpoints)}\nResponse body:\n{{\n  \"text\": \"{payload}\"\n}}"
    templates.append(api_response)

    # 23. Email forwarded chain
    def email_forward(payload, rng):
        names = ["John", "Sarah", "Mike", "HR Department", "IT Support"]
        return f"Forwarded Email:\n---------- Forwarded message ----------\nFrom: {rng.choice(names)}\nDate: 2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}\n\n{payload}"
    templates.append(email_forward)

    # 24. Knowledge base article
    def kb_article(payload, rng):
        ids = ["KB-001234", "KB-005678", "KB-009012", "KB-003456"]
        return f"Knowledge Base Snippet ({rng.choice(ids)}):\n\n{payload}\n\nWas this article helpful?"
    templates.append(kb_article)

    # 25. Translated text
    def translated_text(payload, rng):
        languages = ["Spanish", "French", "German", "Japanese", "Mandarin", "Arabic"]
        return f"Translated text from {rng.choice(languages)} (original):\n\n{payload}"
    templates.append(translated_text)

    return templates


VECTOR_TEMPLATES = _build_vector_templates()


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mt = (row.get("Main Type") or "").strip()
            if mt == "Main Type":
                continue  # skip header artifact
            rows.append(row)
    return rows


def extract_source_payloads(rows: list[dict]) -> list[str]:
    """Extract unique malicious prompts suitable as obfuscation/vector base payloads."""
    seen = set()
    payloads = []
    for row in rows:
        mt = (row.get("Main Type") or "").strip()
        if mt not in SOURCE_CATEGORIES:
            continue
        text = normalize_text(row.get("text", ""))
        if len(text) < 15 or len(text) > 500:
            continue
        fp = fingerprint(text)
        if fp in seen or not fp:
            continue
        seen.add(fp)
        payloads.append(text)
    return payloads


# ═══════════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_obfuscation_samples(
    payloads: list[str],
    existing_fps: set,
    current_count: int,
    target: int,
    rng: random.Random,
) -> list[dict]:
    needed = target - current_count
    if needed <= 0:
        print(f"  Obfuscation already at {current_count} (target {target}). Skipping.")
        return []

    print(f"  Generating {needed} obfuscation samples from {len(payloads)} source payloads...")
    shuffled = payloads[:]
    rng.shuffle(shuffled)

    results = []
    technique_counts = {}
    attempts = 0
    max_attempts = needed * 20

    while len(results) < needed and attempts < max_attempts:
        base = shuffled[attempts % len(shuffled)]
        technique_fn, sub_type = rng.choice(OBFUSCATION_TECHNIQUES)
        tech_name = technique_fn.__name__

        try:
            augmented = technique_fn(base, rng)
        except Exception:
            attempts += 1
            continue

        if not is_high_quality(augmented):
            attempts += 1
            continue

        fp = fingerprint(augmented)
        if not fp or fp in existing_fps:
            attempts += 1
            continue

        existing_fps.add(fp)
        results.append({
            "text": augmented,
            "Main Type": "Evasion & Obfuscation Attacks",
            "Sub-Type": sub_type,
            "Specific Type": tech_name.replace("apply_", ""),
        })
        technique_counts[tech_name] = technique_counts.get(tech_name, 0) + 1
        attempts += 1

    print(f"  Generated {len(results)} obfuscation samples.")
    print(f"  Technique distribution: {technique_counts}")
    return results


def generate_vector_samples(
    payloads: list[str],
    existing_fps: set,
    current_count: int,
    target: int,
    rng: random.Random,
) -> list[dict]:
    needed = target - current_count
    if needed <= 0:
        print(f"  Vector already at {current_count} (target {target}). Skipping.")
        return []

    print(f"  Generating {needed} vector samples from {len(payloads)} source payloads...")
    shuffled = payloads[:]
    rng.shuffle(shuffled)

    results = []
    template_counts = {}
    attempts = 0
    max_attempts = needed * 20

    while len(results) < needed and attempts < max_attempts:
        base = shuffled[attempts % len(shuffled)]
        template_fn = rng.choice(VECTOR_TEMPLATES)
        tname = template_fn.__name__

        try:
            augmented = template_fn(base, rng)
        except Exception:
            attempts += 1
            continue

        if not is_high_quality(augmented):
            attempts += 1
            continue

        fp = fingerprint(augmented)
        if not fp or fp in existing_fps:
            attempts += 1
            continue

        existing_fps.add(fp)
        results.append({
            "text": augmented,
            "Main Type": "Advanced Vector Attacks",
            "Sub-Type": "Indirect Prompt Injection (IPI)",
            "Specific Type": tname,
        })
        template_counts[tname] = template_counts.get(tname, 0) + 1
        attempts += 1

    print(f"  Generated {len(results)} vector samples.")
    print(f"  Template distribution: {template_counts}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# BENIGN VECTOR SAMPLES (benign content in document templates)
# ═══════════════════════════════════════════════════════════════════════════

BENIGN_DOCUMENT_PAYLOADS = [
    # --- Document processing / analysis requests ---
    "Summarize the key findings from this quarterly report.",
    "Extract the action items from these meeting notes.",
    "List the compliance requirements mentioned in this policy document.",
    "What are the main takeaways from this research paper?",
    "Identify the deadlines mentioned in this project proposal.",
    "Summarize the budget allocations in this financial statement.",
    "What deliverables are outlined in this statement of work?",
    "Extract the contact information from this business card scan.",
    "List the agenda items from this board meeting transcript.",
    "What are the recommended next steps in this audit report?",
    "Summarize the changes between version 2.3 and 2.4 in these release notes.",
    "Extract the medication dosages from this prescription document.",
    "What safety procedures are described in this workplace manual?",
    "List the ingredients and nutritional information from this product label.",
    "Identify the key performance indicators in this quarterly review.",
    "What training requirements are specified in this onboarding document?",
    "Summarize the warranty terms from this product documentation.",
    "Extract the shipping details from this order confirmation.",
    "What are the eligibility criteria in this grant application guide?",
    "List the technical specifications from this product datasheet.",
    # --- Business content (emails, memos, reports) ---
    "Please review the attached quarterly earnings summary and provide feedback.",
    "The new employee handbook has been updated with the latest HR policies.",
    "Here is the draft proposal for the client presentation next Tuesday.",
    "The IT department has scheduled maintenance for this weekend.",
    "Updated travel expense guidelines are attached for your review.",
    "Please find the meeting minutes from yesterday's strategy session.",
    "The vendor has submitted their response to our RFP.",
    "Attached is the risk assessment for the new product launch.",
    "The marketing team has finalized the campaign performance report.",
    "Please review the updated service level agreement before signing.",
    "The annual performance review templates have been distributed.",
    "Here are the consolidated customer feedback results from Q3.",
    "The facilities team has completed the office safety inspection report.",
    "Please find the updated organizational chart attached.",
    "The procurement department has submitted the supplier evaluation.",
    "Attached are the training completion certificates for the new hires.",
    "The legal team has reviewed and approved the partnership agreement.",
    "Here is the project status update for sprint 47.",
    "The customer success team has prepared the monthly churn analysis.",
    "Please review the infrastructure capacity planning document.",
    # --- Technical / IT content ---
    "The server migration is scheduled for March 15th during the maintenance window.",
    "Database backup completed successfully. Total size: 42.3 GB.",
    "The SSL certificate for api.company.com expires on April 30th.",
    "Memory usage on prod-server-03 has been trending upward this week.",
    "The CI/CD pipeline completed 847 builds this month with a 96.2% success rate.",
    "Network latency between us-east and eu-west has improved after the CDN update.",
    "The load balancer configuration has been updated to support the new endpoints.",
    "Code coverage for the authentication module increased from 78% to 91%.",
    "The Kubernetes cluster has been scaled to handle the expected holiday traffic.",
    "API response times averaged 145ms across all endpoints this quarter.",
    "The firewall rules have been updated per the latest security audit recommendations.",
    "Disk utilization on the analytics cluster is at 73% capacity.",
    "The automated test suite now includes 2,340 integration tests.",
    "Deployment to staging environment completed without errors.",
    "The data warehouse ETL pipeline processed 12 million records yesterday.",
    # --- HR / Administrative ---
    "The office will be closed on March 21st for the company retreat.",
    "New parking assignments have been posted in the lobby.",
    "The health insurance open enrollment period begins next Monday.",
    "Team lunch is scheduled for Friday at noon in the main conference room.",
    "The building access cards will be reissued next week.",
    "Please submit your timesheets by end of day Thursday.",
    "The company picnic is scheduled for June 15th at Riverside Park.",
    "Updated emergency contact forms are due by the end of the month.",
    "The new recycling program starts next week in all office locations.",
    "Conference room B has been reserved for the workshop on Wednesday.",
    # --- Academic / Research ---
    "This paper examines the effects of urbanization on local biodiversity.",
    "The study found a statistically significant correlation between sleep quality and productivity.",
    "Table 3 shows the distribution of survey responses across age groups.",
    "The literature review covers publications from 2018 to 2024.",
    "Methodology: We used a double-blind randomized controlled trial with 500 participants.",
    "The results indicate a 23% improvement in response time after the intervention.",
    "Figure 7 illustrates the relationship between temperature and crop yield.",
    "The control group showed no significant change over the 12-week period.",
    "Our findings are consistent with previous research by Chen et al. (2022).",
    "The dataset contains 15,000 annotated samples across five categories.",
    # --- Customer service / Support ---
    "Thank you for contacting support. Your ticket number is TKT-78432.",
    "The replacement part has been shipped and should arrive within 3-5 business days.",
    "We have processed your refund. Please allow 5-7 business days for it to appear.",
    "Your subscription has been renewed for another 12 months.",
    "The software update fixes the display issue reported in your previous ticket.",
    "We have escalated your request to the engineering team for further investigation.",
    "Your account has been successfully migrated to the new platform.",
    "The warranty covers manufacturing defects for a period of 24 months.",
    "A technician has been scheduled to visit your location on Thursday between 9-12.",
    "Your feature request has been added to the product roadmap for Q2.",
    # --- Financial / Legal ---
    "The invoice total is $12,450.00 with payment terms of Net 30.",
    "All transactions have been reconciled for the fiscal year ending December 31.",
    "The contract renewal is due on April 1st with the same terms as last year.",
    "Revenue increased 8.3% year-over-year driven by enterprise subscriptions.",
    "The tax filing deadline for Q1 estimated payments is April 15th.",
    "Operating expenses were 12% under budget for the current quarter.",
    "The audit committee has approved the internal controls assessment.",
    "Accounts receivable aging shows 94% of invoices are within 30 days.",
    "The board has approved the capital expenditure budget for the new facility.",
    "Monthly recurring revenue reached $2.1M, up from $1.8M last quarter.",
    # --- General knowledge / Everyday ---
    "What are the best practices for organizing a remote team standup?",
    "How do I convert a PDF to an editable Word document?",
    "What is the recommended format for a professional cover letter?",
    "Can you explain the difference between TCP and UDP protocols?",
    "What are some effective strategies for managing project deadlines?",
    "How do I set up two-factor authentication on my work account?",
    "What is the standard procedure for requesting time off?",
    "Can you recommend some resources for learning data visualization?",
    "What are the steps to file an expense report in the new system?",
    "How do I configure email forwarding for my work account?",
    # --- Mixed realistic requests ---
    "Please translate this customer complaint from Spanish to English.",
    "Organize these interview notes by candidate and summarize each.",
    "Compare the pricing in these two vendor proposals.",
    "Extract the revision history from this document.",
    "What are the access permissions defined in this security policy?",
    "List the milestones and their due dates from this project plan.",
    "Summarize the patient intake form for the doctor's review.",
    "What software licenses are listed in this asset inventory?",
    "Organize the agenda items from this meeting transcript.",
    "Extract the enrollment requirements from this course catalog.",
    "What are the revision dates mentioned in this software changelog?",
    "Summarize the travel itinerary from this booking confirmation.",
    "List the regulatory changes from this compliance update memo.",
    "What medications are listed in this prescription scan?",
    "Extract the nutritional information from this food label.",
    "What are the compliance deadlines in this regulatory filing?",
    "Summarize the customer satisfaction scores from this survey.",
    "List the open issues from this bug tracker export.",
    "What maintenance tasks are scheduled in this facilities report?",
    "Extract the speaker bios from this conference program.",
]


def generate_benign_vector_samples(
    existing_fps: set,
    current_count: int,
    target: int,
    rng: random.Random,
) -> list[dict]:
    needed = target - current_count
    if needed <= 0:
        print(f"  Benign vector already at {current_count} (target {target}). Skipping.")
        return []

    print(f"  Generating {needed} benign vector samples from {len(BENIGN_DOCUMENT_PAYLOADS)} curated payloads...")
    payloads = BENIGN_DOCUMENT_PAYLOADS[:]
    rng.shuffle(payloads)

    results = []
    template_counts = {}
    attempts = 0
    max_attempts = needed * 20

    while len(results) < needed and attempts < max_attempts:
        base = payloads[attempts % len(payloads)]
        template_fn = rng.choice(VECTOR_TEMPLATES)
        tname = template_fn.__name__

        try:
            augmented = template_fn(base, rng)
        except Exception:
            attempts += 1
            continue

        if not is_high_quality(augmented):
            attempts += 1
            continue

        fp = fingerprint(augmented)
        if not fp or fp in existing_fps:
            attempts += 1
            continue

        existing_fps.add(fp)
        results.append({
            "text": augmented,
            "Main Type": "Benign",
            "Sub-Type": "vector_benign",
            "Specific Type": tname,
        })
        template_counts[tname] = template_counts.get(tname, 0) + 1
        attempts += 1

    print(f"  Generated {len(results)} benign vector samples.")
    print(f"  Template distribution: {template_counts}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    rng = random.Random(SEED)

    print("=" * 70)
    print("Training Data Augmentation: Obfuscation & Vector Attack Balancing")
    print("=" * 70)
    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Targets: Obfuscation={TARGET_OBFUSCATION}, Vector={TARGET_VECTOR}")
    print()

    # 1. Load
    rows = load_dataset(INPUT_CSV)
    print(f"Loaded {len(rows)} rows.")

    # 2. Current distribution
    from collections import Counter
    dist = Counter(r.get("Main Type", "").strip() for r in rows)
    print("\nCurrent distribution:")
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # 3. Extract source payloads
    payloads = extract_source_payloads(rows)
    print(f"\nSource payloads available: {len(payloads)}")

    # 4. Existing fingerprints
    existing_fps = {fingerprint(r["text"]) for r in rows if fingerprint(r.get("text", ""))}
    print(f"Existing fingerprints: {len(existing_fps)}")

    # 5. Generate obfuscation
    print("\n--- Obfuscation Augmentation ---")
    obf_count = sum(1 for r in rows if r.get("Main Type", "").strip() == "Evasion & Obfuscation Attacks")
    new_obf = generate_obfuscation_samples(payloads, existing_fps, obf_count, TARGET_OBFUSCATION, rng)

    # 6. Generate vector
    print("\n--- Vector Attack Augmentation ---")
    vec_count = sum(1 for r in rows if r.get("Main Type", "").strip() == "Advanced Vector Attacks")
    new_vec = generate_vector_samples(payloads, existing_fps, vec_count, TARGET_VECTOR, rng)

    # 7. Strip any benign_vector samples (not used in training — hurts recall)
    old_bv = sum(1 for r in rows if r.get("Sub-Type", "").strip() == "vector_benign")
    if old_bv:
        rows = [r for r in rows if r.get("Sub-Type", "").strip() != "vector_benign"]
        print(f"\n  Removed {old_bv} benign_vector samples (not used — hurts attack recall).")

    # 8. Merge
    all_rows = rows + new_obf + new_vec
    rng.shuffle(all_rows)

    # 8. Save
    fieldnames = ["text", "Main Type", "Sub-Type", "Specific Type"]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nSaved {len(all_rows)} rows to {OUTPUT_CSV}")

    # 9. Final distribution
    final_dist = Counter(r.get("Main Type", "").strip() for r in all_rows)
    print("\nFinal distribution:")
    for cat, count in sorted(final_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    total_mal = sum(c for cat, c in final_dist.items() if cat != "Benign")
    print(f"\nTotal: {len(all_rows)} (Benign={final_dist.get('Benign', 0)}, Malicious={total_mal})")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("  1. Regenerate 2-class CSV if needed")
    print("  2. Retrain classifier:  python scripts/training/finetune_deberta_2class_mof.py")
    print("  3. Rebuild RAG index:   cd ../app/backend && python build_rag_index.py")
    print("  4. Re-evaluate:         Run eval datasets through the app")
    print("=" * 70)


if __name__ == "__main__":
    main()
