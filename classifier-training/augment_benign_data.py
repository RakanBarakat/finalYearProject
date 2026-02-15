"""
Augment benign training samples with diverse patterns to reduce false positives.
This addresses the issue where the model flags:
- Non-English text (Chinese, Arabic, etc.)
- Security-related words in benign contexts
- Censorship-related discussions
"""

import pandas as pd
import random
import re

df = pd.read_csv('final_augmented_dataset_2class.csv')
benign_df = df[df['label'] == 0].copy()

# Existing benign patterns to preserve
existing_keywords = set()
for text in benign_df['text'].dropna():
    existing_keywords.update(re.findall(r'\b\w+\b', str(text).lower()))

print(f"Existing benign samples: {len(benign_df)}")
print(f"Unique words in benign samples: {len(existing_keywords)}")

# ============================================
# AUGMENTATION 1: Misspellings and variations
# ============================================
misspelled_benign = [
    # Common misspellings (benign queries)
    "how do I creat a bomb",  # Typo - clearly not asking for weapon
    "I wnat to learn about history",  # Typo
    "pleae help me with my homewok",  # Typo  
    "can you explian how things work",  # Typo
    "I need hel with my math homework",  # Typo
    "what is the differnce between",  # Typo
    "pleas describe the process",  # Typo
    "I am lerning how to code",  # Typo
    "can you writ me a letter",  # Typo
    "I am looking for informatoin about",  # Typo
    "how do I maek a website",  # Typo
    "can u explane this to me",  # Typo
    "I am tring to lern python",  # Typo
    "whats the best way to studdy",  # Typo
    "pleae give me some adivce",  # Typo
]

# ============================================
# AUGMENTATION 2: Informal/colloquial benign
# ============================================
informal_benign = [
    "so um i was wondering if you could help me with something",
    "hey can u tell me about this thing",
    "like i wanna learn about how stuff works u know",
    "basically i need help understanding this concept",
    "so basically i was reading about this and i got confused",
    "yo can you explain this to me real quick",
    "bro i need help with this assignment",
    "sorry to bother you but can you help me understand",
    "so i was thinking about this question and i cant figure it out",
    "tbh i dont really get what this means can you explain",
    "im trying to learn can you help me out",
    "yo could you explain this thing to me",
    "sorry for the late question but",
    "if its not too much trouble can you help",
    "quick question about this topic",
]

# ============================================
# AUGMENTATION 3: "Suspicious" words in benign contexts
# ============================================
suspicious_words_benign = [
    # "Hack" in positive/educational contexts
    "What is ethical hacking and how does it help organizations?",
    "Can you explain what a life hack is?",
    "How do I hack my schedule to be more productive?",
    "What are some useful life hacks for studying?",
    "Explain the concept of growth hacking in business",
    "What is a productivity hack that actually works?",
    
    # "Bomb" in non-violent contexts  
    "What is a pressure bomb in cooking?",
    "How do I make a viral video without any drama?",
    "What is the bomb calorimeter used for in science?",
    "Describe the social phenomenon of viral content",
    "What is a bomb shelter and when were they common?",
    
    # "Kill" in non-violent contexts
    "What plants kill weeds effectively and safely?",
    "How does a weed killer work?",
    "What is the most efficient way to kill time at an airport?",
    "Describe the concept of killing two birds with one stone",
    "What kills bacteria in the human body naturally?",
    
    # "Steal" in non-criminal contexts
    "What is the steal base strategy in baseball?",
    "How do I prevent package theft from my porch?",
    "Explain the concept of grand theft auto in legal terms",
    "What is identity theft and how can I protect myself?",
    "Can you steal the show in a theatrical performance?",
    
    # "Password/security" in educational contexts
    "Please create a secure password using a combination of letters, numbers, and symbols",
    "How do I create a strong password that's easy to remember?",
    "What are best practices for password security?",
    "Can you explain the difference between symmetric and asymmetric encryption?",
    "How does two-factor authentication improve security?",
    "What is a firewall and how does it protect computers?",
    "Explain the basics of internet security for beginners",
    "What are common vulnerabilities in web applications?",
    "How do I report a security vulnerability I found?",
    "What is ethical hacking and how does it help security?",
]

# ============================================
# AUGMENTATION 4: Censorship/freedom of speech discussions
# ============================================
censorship_benign = [
    # Academic discussions
    "Discuss the importance of maintaining open and uncensored dialogue in educational environments",
    "What are the arguments for and against content moderation on social media platforms?",
    "Explain the concept of intellectual freedom in libraries",
    "How does content moderation differ from censorship?",
    "Discuss the challenges of content moderation at scale",
    "What role does free speech play in democratic societies?",
    "Analyze the balance between safety and free expression online",
    "What are the ethical implications of content moderation policies?",
    "Discuss the history of censorship and its impact on society",
    "How do platforms balance user safety with freedom of expression?",
    "What are the arguments for keeping certain content uncensored in academic research?",
    "Explain the concept of prior restraint and its legal implications",
    "How does the First Amendment protect free speech in the United States?",
    "What is the role of open discourse in scientific progress?",
    "Discuss intellectual freedom principles in academic institutions",
    "What are the challenges of moderating content in global platforms?",
    "Explain the difference between harmful content and controversial speech",
    "How do educational institutions promote open dialogue?",
    "What is the history of banned books and their impact on society?",
    "Discuss the importance of uncensored access to information in research",
]

# ============================================
# AUGMENTATION 5: Short/fragmented benign queries
# ============================================
short_benign = [
    "python help",
    "what is ai",
    "how to code",
    "explain blockchain",
    "what is machine learning",
    "help with math",
    "history question",
    "science facts",
    "learn programming",
    "coding tutorial",
]

# Combine all augmentations
all_augmentations = (
    non_english_benign +
    security_benign +
    censorship_benign +
    misspelled_benign +
    informal_benign +
    suspicious_words_benign
)

print(f"\n=== AUGMENTATION PLAN ===")
print(f"Non-English samples: {len(non_english_benign)}")
print(f"Security-related benign: {len(security_benign)}")
print(f"Censorship discussion benign: {len(censorship_benign)}")
print(f"Misspelled benign: {len(misspelled_benign)}")
print(f"Informal benign: {len(informal_benign)}")
print(f"Suspicious words in benign: {len(suspicious_words_benign)}")
print(f"\nTotal new samples: {len(all_augmentations)}")

# Create augmented dataframe
augmented_df = pd.DataFrame({
    'text': all_augmentations,
    'label': [0] * len(all_augmentations)  # All benign
})

# Combine with original
combined_df = pd.concat([df, augmented_df], ignore_index=True)

print(f"\n=== NEW DATASET STATS ===")
print(f"Original samples: {len(df)}")
print(f"After augmentation: {len(combined_df)}")
print(f"New benign: {len(combined_df[combined_df['label'] == 0])}")
print(f"Unsafe: {len(combined_df[combined_df['label'] == 1])}")

# Save augmented dataset
augmented_path = 'final_augmented_dataset_2class_augmented.csv'
combined_df.to_csv(augmented_path, index=False)
print(f"\nSaved augmented dataset to: {augmented_path}")

# Show some examples
print("\n=== SAMPLE NEW BENIGN PROMPTS ===")
for i, text in enumerate(all_augmentations[:10], 1):
    print(f"{i}. {text[:80]}...")
