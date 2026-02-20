"""
Add benign prompts with injection-style trigger words in LEGITIMATE contexts.
Distinct from NotInject: different templates, phrasing, and use cases.
Purpose: Teach the classifier that "ignore", "override", "instructions", etc.
         can appear in harmless queries (document editing, dev tools, tutorials).
"""

import pandas as pd

# Benign prompts with trigger words in legitimate contexts (label=0)
# NOT copied from NotInject - unique templates for document editing, dev, tutorials, etc.

BENIGN_INJECTION_PHRASES = [
    # === IGNORE (document editing, data analysis, clarification) ===
    "Please ignore the typo in my last message and answer the question about Python.",
    "Ignore the formatting and extract the key facts from this document for my summary.",
    "In your analysis, ignore the outliers and focus on the main trend.",
    "Can you ignore the sidebar text and summarize just the main article?",
    "Ignore my previous question if it was unclear; I meant to ask about climate change.",
    "When proofreading, please ignore any comments in the margin.",
    "Ignore the placeholder text and tell me if the structure looks correct.",
    "For the word count, ignore the footnotes and bibliography.",
    "Ignore the draft version and review the final paragraph instead.",
    "Can you ignore the header and just summarize the body of this email?",
    "Ignore the formatting errors and focus on the content.",
    "In your reply, ignore the part about pricing—I already have that.",
    "Please ignore the duplicate section and merge the rest.",
    "Ignore the test data and analyze only the production logs.",
    "When translating, ignore the parenthetical notes.",
    
    # === OVERRIDE (software, settings, config) ===
    "How do I override the default font size in this template?",
    "Can you override the CSS for this component to make it responsive?",
    "What's the override for the default timeout in this API?",
    "How do I override the default behavior in this config file?",
    "Is there a way to override the template for this email?",
    "How can I override the system theme in this app?",
    "What override do I need for the cache settings?",
    "How do I override the default value in this form?",
    "Can you show me how to override the base class method?",
    "What's the override for the default language in the settings?",
    "How do I override the default export format?",
    "Is there an override for the default keyboard shortcut?",
    "How can I override the default margin in the document?",
    "What override do I use for the default port?",
    "How do I override the default encoding for this file?",
    
    # === INSTRUCTIONS (recipes, manuals, tutorials) ===
    "These assembly instructions are unclear—can you simplify them?",
    "What do the instructions say about the battery installation?",
    "The instructions are confusing; can you break them down step by step?",
    "Can you rewrite these instructions for a beginner?",
    "The instructions mention a safety warning—what does it say?",
    "These cooking instructions are too vague; can you clarify?",
    "What are the instructions for the setup wizard?",
    "The instructions are in a different language—can you summarize?",
    "Can you extract the key instructions from this manual?",
    "The instructions are outdated; what's the modern approach?",
    "What do the instructions say about maintenance?",
    "Can you translate these instructions into plain English?",
    "The instructions are contradictory—which one is correct?",
    "What instructions are missing from this guide?",
    "Can you add instructions for troubleshooting?",
    
    # === SYSTEM PROMPT (programming, dev docs) ===
    "What is a system prompt in the context of chatbot development?",
    "Explain how system prompts work in LLM applications.",
    "How do system prompts differ from user prompts in API design?",
    "What's the best practice for writing system prompts?",
    "Can you explain system prompts in the context of ChatGPT?",
    "What role do system prompts play in prompt engineering?",
    "How do system prompts affect model behavior?",
    "What is the system prompt in this code example?",
    "Explain the concept of system prompts for my presentation.",
    "How are system prompts used in RAG applications?",
    "What's the difference between system and user prompts?",
    "Can you explain system prompts in simple terms?",
    "How do I document system prompts in my project?",
    "What are the limitations of system prompts?",
    "How do system prompts interact with few-shot examples?",
    
    # === BYPASS (dev, testing, troubleshooting) ===
    "How do I bypass the login for local development testing?",
    "Is there a bypass for the cache when debugging?",
    "How can I bypass the redirect for testing?",
    "What's the bypass for the rate limit in dev mode?",
    "How do I bypass the SSL check for local testing?",
    "Is there a bypass for the captcha in the test environment?",
    "How can I bypass the auth for unit tests?",
    "What's the bypass for the firewall in the sandbox?",
    "How do I bypass the validation for seed data?",
    "Is there a bypass for the timeout in the staging env?",
    "How can I bypass the middleware for this specific route?",
    "What's the bypass for the proxy in local dev?",
    "How do I bypass the CDN for debugging?",
    "Is there a bypass for the cache when testing?",
    "How can I bypass the redirect loop for testing?",
    
    # === RESTRICTIONS (API docs, compliance, features) ===
    "What are the restrictions on this API's free tier?",
    "Explain the restrictions for this feature in the docs.",
    "Are there restrictions on file size for uploads?",
    "What restrictions apply to the trial version?",
    "What are the rate limit restrictions for this endpoint?",
    "Explain the geographic restrictions for this service.",
    "What restrictions apply to the export format?",
    "Are there restrictions on the number of requests?",
    "What are the licensing restrictions for this library?",
    "Explain the restrictions for the sandbox environment.",
    "What restrictions apply to the free plan?",
    "Are there restrictions on the data retention period?",
    "What are the usage restrictions for this API?",
    "Explain the restrictions for the community edition.",
    "What restrictions apply to the beta feature?",
    
    # === PREVIOUS (academic, iterative, context) ===
    "Referring to the previous example, how would the formula change?",
    "Unlike the previous version, what's new in this release?",
    "Building on the previous point, can you expand on that?",
    "Compared to the previous draft, what did you change?",
    "In light of the previous feedback, here's the revised version.",
    "The previous section was unclear—can you rephrase?",
    "Following the previous discussion, what's the next step?",
    "As mentioned in the previous email, when is the deadline?",
    "The previous answer was helpful—one more question.",
    "Compared to the previous approach, is this faster?",
    "Building on the previous chapter, what comes next?",
    "The previous version had a bug—is it fixed now?",
    "Referring to the previous slide, what's the key takeaway?",
    "Unlike the previous model, what improvements does this have?",
    "Following the previous instructions, what's the next step?",
    
    # === REVEAL (UI, docs, formatting) ===
    "Can you reveal the hidden formatting in this document?",
    "How do I reveal the structure of this HTML page?",
    "Reveal the outline of the essay so I can check the flow.",
    "Can you reveal the hidden columns in this spreadsheet?",
    "How do I reveal the markdown source in this editor?",
    "Reveal the table of contents for this document.",
    "Can you reveal the metadata in this file?",
    "How do I reveal the comments in this PDF?",
    "Reveal the revision history for this document.",
    "Can you reveal the structure of this JSON?",
    "How do I reveal the alt text for these images?",
    "Reveal the headings hierarchy in this article.",
    "Can you reveal the hidden characters in this text?",
    "How do I reveal the source of this quote?",
    "Reveal the outline of the presentation for me.",
]

def main():
    # Load existing 2-class dataset
    df = pd.read_csv("final_augmented_dataset_2class.csv")
    df["text"] = df["text"].astype(str)
    
    original_count = len(df)
    
    # Create new benign samples
    new_rows = pd.DataFrame({
        "text": BENIGN_INJECTION_PHRASES,
        "label": [0] * len(BENIGN_INJECTION_PHRASES)  # 0 = Benign
    })
    
    # Append (avoid duplicates)
    existing_texts = set(df["text"].str.strip().str.lower())
    to_add = []
    for _, row in new_rows.iterrows():
        t = str(row["text"]).strip()
        if t.lower() not in existing_texts:
            to_add.append({"text": t, "label": 0})
            existing_texts.add(t.lower())
    
    if to_add:
        df_new = pd.DataFrame(to_add)
        df_combined = pd.concat([df, df_new], ignore_index=True)
        df_combined.to_csv("final_augmented_dataset_2class.csv", index=False)
        print(f"Added {len(to_add)} benign injection-phrase samples")
        print(f"Original: {original_count} rows")
        print(f"Final: {len(df_combined)} rows")
    else:
        print("No new samples to add (all already exist)")
        print(f"Total: {original_count} rows")
    print(f"\nDataset saved to: final_augmented_dataset_2class.csv")


if __name__ == "__main__":
    main()
