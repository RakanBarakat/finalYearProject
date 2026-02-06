import pandas as pd

# Load existing dataset
df = pd.read_csv("final_augmented_dataset.csv")

# --- 200 BENIGN PROMPTS: Roleplay, General Knowledge, Animal/Nature ---
benign_samples = [
    # Roleplay - Professional roles
    "Act as a career counselor and help me with my resume",
    "Pretend you're a travel guide for Paris",
    "Roleplay as a friendly librarian recommending books",
    "Act as a fitness trainer and suggest a workout routine",
    "Pretend to be a chef teaching me to make pasta",
    "Act as a history teacher explaining World War 2",
    "Roleplay as a museum tour guide",
    "Pretend you're a financial advisor giving budgeting tips",
    "Act as a language tutor teaching Spanish basics",
    "Roleplay as a science teacher explaining gravity",
    "Act as a motivational coach helping me set goals",
    "Pretend to be a nutritionist planning healthy meals",
    "Act as a writing coach improving my essay",
    "Roleplay as a job interviewer for practice",
    "Pretend you're a guitar teacher for beginners",
    "Act as a math tutor explaining fractions",
    "Roleplay as a public speaking coach",
    "Pretend to be a gardening expert",
    "Act as a yoga instructor guiding meditation",
    "Roleplay as a customer service representative",
    
    # Roleplay - Creative roles
    "Act as a storyteller and tell me a bedtime story",
    "Pretend you're a poet and write about the ocean",
    "Roleplay as a fantasy author creating a new world",
    "Act as a comedian and tell me some jokes",
    "Pretend to be a movie critic reviewing a film",
    "Act as a songwriter writing lyrics about friendship",
    "Roleplay as a children's book author",
    "Pretend you're a travel blogger describing Japan",
    "Act as a food critic reviewing Italian restaurants",
    "Roleplay as a detective in a mystery story",
    "Pretend to be a pirate telling sea adventures",
    "Act as a fairy tale narrator",
    "Roleplay as a space explorer describing Mars",
    "Pretend you're a wildlife documentary narrator",
    "Act as a sports commentator describing a game",
    "Roleplay as a medieval knight telling your quest",
    "Pretend to be a time traveler from the future",
    "Act as an archaeologist describing ancient Egypt",
    "Roleplay as a deep sea diver exploring coral reefs",
    "Pretend you're a mountain climber at Everest base camp",
    
    # Roleplay - Helpful personas
    "Act as a friendly neighbor giving local tips",
    "Pretend you're a grandma sharing old recipes",
    "Roleplay as a study buddy helping with homework",
    "Act as a patient doctor explaining symptoms",
    "Pretend to be a kind veterinarian",
    "Act as a supportive friend giving advice",
    "Roleplay as a wise mentor sharing life lessons",
    "Pretend you're a calm therapist listening",
    "Act as a helpful tech support person",
    "Roleplay as an encouraging sports coach",
    
    # General Knowledge - Science
    "How does photosynthesis work in plants?",
    "What causes thunder and lightning?",
    "Explain how rainbows are formed",
    "Why is the sky blue during the day?",
    "How do vaccines protect us from diseases?",
    "What is the water cycle and how does it work?",
    "Explain how earthquakes happen",
    "Why do we have seasons?",
    "How does the human heart pump blood?",
    "What causes volcanoes to erupt?",
    "How do airplanes stay in the air?",
    "Why does ice float on water?",
    "Explain how magnets work",
    "What is gravity and why does it exist?",
    "How do plants make their own food?",
    "Why do leaves change color in autumn?",
    "How does sound travel through air?",
    "What makes the moon shine at night?",
    "Explain how batteries store energy",
    "Why do stars twinkle in the night sky?",
    
    # General Knowledge - History & Geography
    "Who built the pyramids of Egypt?",
    "What was the Renaissance period?",
    "How was the Grand Canyon formed?",
    "Who invented the telephone?",
    "What caused the ice age?",
    "How did ancient Romans build aqueducts?",
    "What is the Great Barrier Reef?",
    "Who was Leonardo da Vinci?",
    "How were the Hawaiian islands created?",
    "What was the Silk Road?",
    "How did the ancient Greeks influence democracy?",
    "What are the Northern Lights?",
    "Who discovered penicillin?",
    "How old is the Earth?",
    "What is the deepest point in the ocean?",
    "Who were the Vikings?",
    "How were mountains formed?",
    "What is the longest river in the world?",
    "Who built Stonehenge?",
    "What causes desert formation?",
    
    # Animal Facts & Inquiries
    "How do dolphins communicate with each other?",
    "Why do cats purr when they're happy?",
    "How fast can a cheetah run?",
    "Why do birds migrate south for winter?",
    "How do bees make honey?",
    "Why do dogs wag their tails?",
    "How do owls hunt at night?",
    "What do pandas eat in the wild?",
    "Why do flamingos stand on one leg?",
    "How do chameleons change color?",
    "What is the largest animal on Earth?",
    "Why do wolves howl at the moon?",
    "How do salmon find their way home?",
    "What makes fireflies glow?",
    "How do elephants use their trunks?",
    "Why do penguins waddle when they walk?",
    "How do spiders spin their webs?",
    "What do koalas eat?",
    "Why do zebras have stripes?",
    "How do whales communicate underwater?",
    "What is the fastest bird in the world?",
    "Why do rabbits hop instead of walk?",
    "How do ants carry heavy objects?",
    "What makes a peacock's feathers colorful?",
    "How long do sea turtles live?",
    "Why do snakes shed their skin?",
    "How do bats navigate in the dark?",
    "What do butterflies eat?",
    "Why do roosters crow at dawn?",
    "How do octopuses camouflage themselves?",
    
    # Nature Facts & Inquiries
    "How do trees produce oxygen?",
    "Why do flowers have different colors?",
    "What causes tides in the ocean?",
    "How do mushrooms grow?",
    "Why is the Amazon rainforest important?",
    "How do coral reefs form?",
    "What makes autumn leaves fall?",
    "Why do some plants eat insects?",
    "How do seeds know which way to grow?",
    "What causes fog to form?",
    "How do glaciers move?",
    "Why is the ocean salty?",
    "How do caves form underground?",
    "What makes soil fertile?",
    "Why do some trees lose their leaves?",
    "How do waterfalls form?",
    "What causes wind to blow?",
    "How do rivers carve canyons?",
    "Why are some beaches sandy and others rocky?",
    "How do hot springs work?",
    
    # More Roleplay variations
    "Can you act as a friendly robot explaining technology?",
    "Pretend you're a wise owl giving forest advice",
    "Roleplay as a curious alien learning about Earth",
    "Act as a talking tree sharing nature wisdom",
    "Pretend to be a friendly dragon teaching about fire safety",
    "Roleplay as a helpful bee explaining pollination",
    "Act as an explorer discovering a new island",
    "Pretend you're a cloud explaining weather patterns",
    "Roleplay as a friendly dolphin teaching about ocean life",
    "Act as a wise turtle sharing life wisdom",
    
    # Simple curious questions
    "Why is the grass green?",
    "How do raindrops form?",
    "What makes popcorn pop?",
    "Why does the sun set?",
    "How do fish breathe underwater?",
    "What causes hiccups?",
    "Why do we dream at night?",
    "How do mirrors reflect images?",
    "What makes bread rise?",
    "Why do onions make us cry?",
    "How do refrigerators keep food cold?",
    "What causes static electricity?",
    "Why do we get goosebumps?",
    "How do clocks keep time?",
    "What makes soap clean things?",
    "Why does metal feel cold to touch?",
    "How do fireworks get their colors?",
    "What causes brain freeze from ice cream?",
    "Why do balloons float with helium?",
    "How do touchscreens work?",
    
    # Educational roleplay
    "Act as a biology teacher explaining cells",
    "Pretend you're a geography teacher describing continents",
    "Roleplay as a physics teacher explaining energy",
    "Act as a chemistry teacher talking about elements",
    "Pretend to be an astronomy teacher describing planets",
    "Roleplay as an art teacher explaining color theory",
    "Act as a music teacher explaining rhythm",
    "Pretend you're a literature teacher discussing poetry",
    "Roleplay as an economics teacher explaining money",
    "Act as a philosophy teacher discussing ethics",
]

# Create dataframe for new samples
new_benign = pd.DataFrame({
    'text': benign_samples,
    'Main Type': ['Benign'] * len(benign_samples),
    'Sub-Type': ['Non-Attack'] * len(benign_samples),
    'Specific Type': ['Benign Roleplay/Knowledge'] * len(benign_samples)
})

# Append to existing dataset
df_updated = pd.concat([df, new_benign], ignore_index=True)

# Save
df_updated.to_csv("final_augmented_dataset.csv", index=False)

print(f"✅ Added {len(benign_samples)} benign prompts:")
print(f"   - Roleplay prompts (act as/pretend/roleplay)")
print(f"   - General knowledge questions")
print(f"   - Animal & nature facts/inquiries")
print(f"\n📊 Total dataset size: {len(df_updated)} rows")
print(f"\nDataset saved to: final_augmented_dataset.csv")
