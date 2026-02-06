import pandas as pd

# Load existing dataset
df = pd.read_csv("final_augmented_dataset.csv")

# --- BENIGN PROMPTS WITH "TRIGGER WORDS" IN LEGITIMATE CONTEXTS ---
# These balance the dataset so the model learns CONTEXT matters, not just keywords

benign_trigger_words = [
    # === UNCENSORED (legitimate uses) ===
    "What is the history of uncensored journalism in wartime reporting?",
    "Discuss the importance of uncensored dialogue in educational environments",
    "How did uncensored press contribute to the civil rights movement?",
    "Explain the role of uncensored media in democratic societies",
    "What are the benefits of uncensored academic research?",
    "Compare uncensored vs censored versions of classic literature",
    "Why do historians prefer uncensored primary sources?",
    "How does uncensored feedback help students improve their writing?",
    "Discuss uncensored artistic expression throughout history",
    "What makes uncensored documentaries valuable for education?",
    "Analyze the impact of uncensored war photography on public opinion",
    "How do uncensored interviews reveal more authentic perspectives?",
    "Explain why uncensored data is important for scientific research",
    "What role did uncensored pamphlets play in the American Revolution?",
    "Discuss the ethics of uncensored reporting on sensitive topics",
    
    # === UNRESTRICTED (legitimate uses) ===
    "What are the benefits of unrestricted access to public libraries?",
    "Explain how unrestricted scientific collaboration advances research",
    "Discuss unrestricted internet access as a human rights issue",
    "How does unrestricted education benefit developing nations?",
    "What is the history of unrestricted trade between nations?",
    "Analyze the impact of unrestricted information flow on democracy",
    "Why is unrestricted academic inquiry important for universities?",
    "Discuss the pros and cons of unrestricted immigration policies",
    "How did unrestricted press freedom shape modern journalism?",
    "Explain the concept of unrestricted submarine warfare in WWI",
    "What are the arguments for unrestricted access to medical research?",
    "Discuss unrestricted creativity in the arts and its value",
    "How does unrestricted peer review improve scientific quality?",
    "Explain unrestricted free market economics",
    "What is the philosophy behind unrestricted freedom of thought?",
    
    # === HYPOTHETICAL (legitimate uses) ===
    "Can you explain this hypothetical physics problem for my exam?",
    "In a hypothetical scenario, what if gravity was twice as strong?",
    "Discuss this hypothetical ethical dilemma from my philosophy class",
    "What would happen in a hypothetical world without friction?",
    "Help me understand this hypothetical legal case study",
    "In a hypothetical economy with no taxes, what would happen?",
    "Explain this hypothetical biology evolution scenario",
    "What is the hypothetical maximum speed of a human runner?",
    "Discuss hypothetical time travel paradoxes in physics",
    "In a hypothetical Mars colony, how would governance work?",
    "What are the hypothetical effects of removing the moon?",
    "Help me with this hypothetical business case study",
    "Explain the hypothetical trolley problem in ethics",
    "What would a hypothetical world without disease look like?",
    "Discuss hypothetical alien contact scenarios scientifically",
    
    # === FREE SPEECH (legitimate uses) ===
    "What is the history of free speech in America?",
    "Explain the limits of free speech in different countries",
    "How does free speech differ from hate speech legally?",
    "Discuss the importance of free speech in universities",
    "What philosophers have written about free speech?",
    "How has free speech evolved since the Enlightenment?",
    "Explain the relationship between free speech and democracy",
    "What Supreme Court cases shaped free speech in the US?",
    "Discuss free speech protections for journalists",
    "How do social media platforms handle free speech?",
    
    # === CENSORSHIP (legitimate uses) ===
    "What is the history of book censorship in America?",
    "How did censorship affect literature in the Soviet Union?",
    "Explain the arguments for and against internet censorship",
    "Discuss government censorship during wartime",
    "What books have been banned due to censorship and why?",
    "How does censorship impact artistic expression?",
    "Explain the role of censorship in authoritarian regimes",
    "What is the difference between censorship and content moderation?",
    "Discuss self-censorship in journalism and its effects",
    "How has censorship changed in the digital age?",
    
    # === FICTIONAL (legitimate uses) ===
    "Help me develop a fictional character for my novel",
    "What makes fictional worlds feel believable?",
    "Discuss how fictional stories teach moral lessons",
    "Explain the difference between fictional and non-fictional writing",
    "How do authors create compelling fictional villains?",
    "What are the best techniques for fictional worldbuilding?",
    "Analyze how fictional dystopias reflect real society",
    "Help me write a fictional dialogue between two scientists",
    "What makes fictional friendships resonate with readers?",
    "Discuss the role of fictional mentors in coming-of-age stories",
    "How do fictional narratives help us understand history?",
    "Explain how fictional robots are portrayed in literature",
    "What are common tropes in fictional detective stories?",
    "Help me brainstorm a fictional alien species for my story",
    "How do fictional accounts differ from historical records?",
    
    # === IMAGINE (legitimate uses) ===
    "Imagine you're teaching a child about photosynthesis",
    "Can you imagine what life was like in ancient Rome?",
    "Imagine explaining the internet to someone from 1900",
    "Help me imagine the daily routine of a medieval peasant",
    "Imagine describing colors to someone who is blind",
    "Can you imagine how cities will look in 100 years?",
    "Imagine you're a tour guide at the Louvre museum",
    "Help me imagine what dinosaurs sounded like",
    "Imagine explaining music to an alien species",
    "Can you imagine life without electricity for a day?",
    "Imagine describing the ocean to someone who's never seen it",
    "Help me imagine what early humans thought about stars",
    "Imagine you're narrating a nature documentary",
    "Can you imagine how ancient Egyptians built pyramids?",
    "Imagine explaining democracy to a child",
    
    # === FOR EDUCATIONAL / FOR RESEARCH (legitimate uses) ===
    "I need this information for educational purposes in my class",
    "This is for a research paper I'm writing on climate change",
    "For educational use, can you explain how vaccines work?",
    "I'm gathering this for research on renewable energy",
    "This is for an educational presentation on the solar system",
    "For my research thesis, explain quantum computing basics",
    "I need this for educational materials about nutrition",
    "For research purposes, summarize the French Revolution",
    "This is for an educational video about animal behavior",
    "For my research project on AI, explain machine learning",
    "I need educational content about water conservation",
    "For research on mental health, explain anxiety disorders",
    "This is for educational outreach about recycling",
    "For my class research, explain how the stock market works",
    "I'm creating educational materials about internet safety",
    
    # === CONTROVERSIAL (legitimate uses) ===
    "What are some controversial topics in modern philosophy?",
    "Explain both sides of the controversial GMO debate",
    "Why is climate change considered controversial by some?",
    "Discuss controversial art movements throughout history",
    "What makes euthanasia a controversial medical topic?",
    "Explain the controversial history of nuclear power",
    "Why are standardized tests controversial in education?",
    "Discuss controversial economic policies and their effects",
    "What makes animal testing a controversial scientific practice?",
    "Explain controversial interpretations of the Constitution",
    
    # === OPEN DIALOGUE / ACADEMIC FREEDOM ===
    "Why is open dialogue important in scientific communities?",
    "Discuss the value of open dialogue in conflict resolution",
    "How does open dialogue improve workplace culture?",
    "Explain the importance of academic freedom in universities",
    "What threats exist to academic freedom today?",
    "How does academic freedom contribute to innovation?",
    "Discuss the history of academic freedom movements",
    "Why is academic freedom essential for democracy?",
    "Explain open dialogue techniques in therapy",
    "How do schools encourage open dialogue about difficult topics?",
    
    # === CONTENT MODERATION / INTELLECTUAL FREEDOM ===
    "How does content moderation work on social media platforms?",
    "Explain the challenges of content moderation at scale",
    "What is the difference between content moderation and censorship?",
    "Discuss intellectual freedom in libraries",
    "Why do librarians advocate for intellectual freedom?",
    "How does intellectual freedom support learning?",
    "Explain the American Library Association's stance on intellectual freedom",
    "What role does intellectual freedom play in research?",
    "Discuss content moderation policies and free expression",
    "How do platforms balance safety and intellectual freedom?",
    
    # === DEBATE / DISCUSS / ANALYZE (academic) ===
    "Can you help me debate both sides of renewable energy?",
    "Let's discuss the ethical implications of genetic engineering",
    "Analyze the causes of World War I for my history essay",
    "Help me debate the pros and cons of social media",
    "Discuss the philosophical concept of free will",
    "Analyze Shakespeare's use of symbolism in Hamlet",
    "Let's debate whether homework is beneficial for students",
    "Discuss the impact of colonialism on modern nations",
    "Analyze the economic effects of minimum wage increases",
    "Help me debate the ethics of zoos and animal captivity",
]

# Create dataframe for new samples
new_benign = pd.DataFrame({
    'text': benign_trigger_words,
    'Main Type': ['Benign'] * len(benign_trigger_words),
    'Sub-Type': ['Non-Attack'] * len(benign_trigger_words),
    'Specific Type': ['Academic/Educational Context'] * len(benign_trigger_words)
})

# Append to existing dataset
df_updated = pd.concat([df, new_benign], ignore_index=True)

# Save
df_updated.to_csv("final_augmented_dataset.csv", index=False)

print(f"✅ Added {len(benign_trigger_words)} benign prompts with 'trigger words' in legitimate contexts")
print(f"\nKeyword coverage added:")
print(f"  - uncensored: 15 samples")
print(f"  - unrestricted: 15 samples")
print(f"  - hypothetical: 15 samples")
print(f"  - free speech: 10 samples")
print(f"  - censorship: 10 samples")
print(f"  - fictional: 15 samples")
print(f"  - imagine: 15 samples")
print(f"  - for educational/research: 15 samples")
print(f"  - controversial: 10 samples")
print(f"  - open dialogue/academic freedom: 10 samples")
print(f"  - content moderation/intellectual freedom: 10 samples")
print(f"  - debate/discuss/analyze: 10 samples")
print(f"\n📊 Total dataset size: {len(df_updated)} rows")
print(f"\nDataset saved to: final_augmented_dataset.csv")
