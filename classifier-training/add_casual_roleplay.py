import pandas as pd

# Load existing dataset
df = pd.read_csv("final_augmented_dataset.csv")

# --- 100 BENIGN CASUAL ROLEPLAY/PERSONA PROMPTS ---
# 50 short casual prompts + 50 with more context

benign_casual_roleplay = [
    # SHORT CASUAL PROMPTS (50)
    # Pretend to be a person - casual chat
    "pretend ur name is david, tell me about your day and how you build apps",
    "pretend your name is sarah and tell me about your favorite hobbies",
    "act like your name is mike and describe your morning routine",
    "roleplay as someone named emma who loves cooking",
    "pretend you're alex and tell me about your job as a teacher",
    "act as if your name is john and you work in a coffee shop",
    "pretend to be lisa and tell me about your weekend plans",
    "your name is tom, tell me about your favorite movies",
    "act like you're rachel and describe your pet dog",
    "pretend you're chris and talk about your favorite music",
    
    # Short persona conversations
    "imagine you're a college student named jake",
    "pretend you're a baker named maria",
    "act as a gardener named bob",
    "roleplay as a photographer named anna",
    "pretend you're a musician named carlos",
    "act like you're a nurse named jenny",
    "imagine you're a writer named sam",
    "pretend to be a chef named marco",
    "act as an artist named sophie",
    "roleplay as a teacher named mr johnson",
    
    # Short day in the life
    "pretend you're a regular person, tell me about your commute",
    "act like someone who just got home from work",
    "roleplay as a parent picking up kids from school",
    "pretend you're a student who just finished an exam",
    "act as someone who just came back from the gym",
    
    # Short hobby-based
    "pretend you love hiking, tell me about trails",
    "act as someone who plays guitar",
    "roleplay as a person who loves video games",
    "pretend you're into photography",
    "act like someone who does yoga",
    
    # Short friendly chats
    "hey pretend you're my friend catching up over coffee",
    "act like we're old friends, what's new with you",
    "pretend we're neighbors chatting",
    "roleplay as a friendly coworker at lunch",
    "act like you're my study buddy",
    
    # Short job chats
    "pretend you work in tech and tell me what you do",
    "act like you're a freelance designer",
    "roleplay as someone who works from home",
    "pretend you're a small business owner",
    "act as a software developer",
    
    # Short personas
    "be a friendly person named pat and chat with me",
    "act as someone named jordan who loves science",
    "pretend to be taylor who works at a library",
    "roleplay as casey who's really into sports",
    "be someone named morgan who enjoys nature walks",
    "act like riley who's passionate about music",
    "pretend you're jamie who loves trying new restaurants",
    "roleplay as drew who collects vintage items",
    "be someone named quinn who does volunteer work",
    "act as avery who's into fitness and wellness",
    
    # LONGER CONTEXTUAL PROMPTS (50)
    # Detailed persona with backstory
    "I'm writing a story and need help with dialogue. Can you pretend to be a character named David who is a 30-year-old software developer living in Seattle? He loves hiking and building mobile apps. Tell me about a typical day in his life.",
    "For a creative writing exercise, please roleplay as Sarah, a pastry chef who owns a small bakery in a cozy neighborhood. She learned to bake from her grandmother. Describe how she starts her morning at the shop.",
    "I'm practicing conversation skills. Can you act as Mike, a friendly neighbor who just moved in next door? He works as an accountant but his real passion is woodworking. Let's have a casual chat about the neighborhood.",
    "Help me with a story I'm writing. Pretend you're Emma, a veterinarian who specializes in exotic animals. She volunteers at the local animal shelter on weekends. What's the most interesting animal she's treated?",
    "I'm doing a class project on different careers. Can you roleplay as Alex, a high school history teacher who makes learning fun with interactive activities? Describe how you engage students.",
    
    # Detailed everyday scenarios
    "For my English practice, can you pretend to be someone named John who works at a local coffee shop? He knows all the regular customers by name and their usual orders. Tell me about your favorite customers.",
    "I'm writing a blog post about work-life balance. Pretend you're Lisa, a working mom who manages to find time for her hobbies. How do you balance everything while still making time for yourself?",
    "Help me understand different perspectives. Act as Tom, a retired teacher who now spends his time volunteering at the community center and watching classic films. What life advice would you share?",
    "I'm creating a podcast about everyday heroes. Roleplay as Rachel, a physical therapist who helps people recover from injuries. What's the most rewarding part of your job?",
    "For a school assignment, pretend you're Chris, a music producer who works from a home studio. How did you get started and what's your creative process like?",
    
    # Detailed hobby discussions
    "I want to learn about hiking. Can you pretend to be an experienced hiker named Jake who has explored trails all over the country? Share your favorite trail and why it's special to you.",
    "I'm interested in baking. Roleplay as Maria, a home baker who started a small Instagram page sharing her recipes. What's your signature dish and how did you develop it?",
    "Help me with a gardening project. Act as Bob, someone who has been gardening for 20 years and grows his own vegetables. What advice would you give to a beginner?",
    "I'm writing about creative hobbies. Pretend you're Anna, a photographer who specializes in nature and wildlife photography. Describe your most memorable photo shoot.",
    "For a music appreciation class, roleplay as Carlos, a guitarist who plays in a local jazz band on weekends while working a regular job. How do you balance both?",
    
    # Detailed professional scenarios
    "I'm researching nursing careers. Can you act as Jenny, a pediatric nurse who has been working at a children's hospital for 10 years? What made you choose this specialty?",
    "Help me understand the writing process. Pretend you're Sam, an author who writes mystery novels and just published your third book. How do you develop your plot ideas?",
    "I'm curious about the restaurant industry. Roleplay as Marco, an Italian chef who trained in Rome and now runs his own restaurant in America. What's your philosophy on cooking?",
    "For an art class discussion, act as Sophie, a painter who creates abstract art inspired by emotions and music. Walk me through how you approach a new canvas.",
    "I want to understand teaching better. Pretend you're Mr. Johnson, a math teacher who uses games and real-world examples to make algebra fun. Share a successful lesson plan.",
    
    # Detailed life experience chats
    "I'm moving to a new city soon. Can you pretend to be someone named Alex who relocated from a small town to a big city last year? Share your experience and tips for adjusting.",
    "I just started a new job. Roleplay as Jordan, someone who switched careers from finance to graphic design at age 35. How did you manage the transition?",
    "I'm thinking about adopting a pet. Act as Taylor, someone who adopted a rescue dog two years ago. Tell me about the experience and any advice you'd give.",
    "I want to learn a new language. Pretend you're Casey, someone who taught themselves Spanish using apps and online resources. How did you stay motivated?",
    "For a personal development project, roleplay as Morgan, someone who took up painting as a hobby during the pandemic and now sells art online. How did that journey unfold?",
    
    # Detailed friendly conversations
    "Let's practice casual conversation. Pretend we're two friends named Pat and me, meeting for coffee after not seeing each other for six months. Catch me up on what's been happening in your life.",
    "I'm working on social skills. Act as Riley, a friendly person I just met at a community gardening event. Let's have a natural conversation about plants and the neighborhood.",
    "Help me practice small talk. Roleplay as Jamie, someone sitting next to me on a long flight. We start chatting about travel destinations and favorite vacation memories.",
    "For a communication class, pretend you're Drew, a fellow attendee at a book club meeting. We're discussing our favorite books and why we joined the club.",
    "I want to practice networking. Act as Quinn, a fellow professional I meet at an industry conference. Let's have a friendly conversation about our careers and interests.",
    
    # Detailed creative scenarios
    "I'm working on a creative writing project. Can you pretend to be Avery, a yoga instructor who started teaching after overcoming a health challenge? Share your journey to wellness.",
    "For a documentary I'm making, roleplay as someone named Blake who runs a community garden in an urban neighborhood. Tell me why you started it and its impact.",
    "Help me with character development. Act as Charlie, a librarian who secretly writes poetry and just got their first poem published. How does it feel?",
    "I'm writing a feel-good story. Pretend you're Dana, a retiree who learned to play piano at 65 and now performs at local nursing homes. Share your experience.",
    "For a podcast about passion projects, roleplay as Elliott, someone who builds furniture from reclaimed wood as a hobby. How did you get into it?",
    
    # Detailed mentor-style conversations
    "I'm starting a small business. Can you pretend to be Frankie, someone who successfully runs an Etsy shop selling handmade jewelry? Share your tips for getting started.",
    "I want advice on work-life balance. Act as Gray, a parent who works remotely and manages to be present for family while excelling at work. What's your secret?",
    "Help me understand freelancing. Roleplay as Harper, a freelance writer who left a corporate job to pursue their passion. Walk me through your typical workday.",
    "I'm exploring career options. Pretend you're Indigo, a UX designer who transitioned from teaching. How did you make that change and what do you love about it?",
    "For a mentorship program, act as Jules, an experienced project manager who helps junior team members grow. What advice would you give someone starting out?",
    
    # Detailed interest-based conversations  
    "I want to learn about astronomy. Pretend you're Kelly, an amateur astronomer who has been stargazing for years and hosts neighborhood viewing parties. What got you interested?",
    "Help me appreciate history. Roleplay as Logan, a history enthusiast who visits historical sites and reads primary sources for fun. What period fascinates you most?",
    "I'm curious about wildlife. Act as Murphy, a nature photographer and bird watcher who can identify hundreds of species. Tell me about an amazing wildlife encounter.",
    "For an environmental project, pretend you're Noel, someone who lives a zero-waste lifestyle and teaches others how to reduce their impact. How did you start?",
    "I want to get into sports. Roleplay as Owen, a recreational soccer player who joined an adult league after never playing as a kid. How was that experience?",
]

# Create dataframe for new samples
new_benign = pd.DataFrame({
    'text': benign_casual_roleplay,
    'Main Type': ['Benign'] * len(benign_casual_roleplay),
    'Sub-Type': ['Non-Attack'] * len(benign_casual_roleplay),
    'Specific Type': ['Casual Roleplay Chat'] * len(benign_casual_roleplay)
})

# Append to existing dataset
df_updated = pd.concat([df, new_benign], ignore_index=True)

# Save
df_updated.to_csv("final_augmented_dataset.csv", index=False)

print(f"✅ Added {len(benign_casual_roleplay)} benign casual roleplay prompts")
print(f"\n📊 Total dataset size: {len(df_updated)} rows")
print(f"\nDataset saved to: final_augmented_dataset.csv")
