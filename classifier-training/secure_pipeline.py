import torch
import requests
import json
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your fine-tuned model (ensure this matches where you saved it)
GUARD_MODEL_PATH = "./final_security_model_3way" 

# Ollama Settings
OLLAMA_URL = "http://localhost:11434/api/generate"
TARGET_LLM = "dolphin-llama3:8b"  # Your installed model

# Thresholds (Optional: Only block if confidence is high?)
# For now, we block everything that isn't Benign (Label 0).
CONFIDENCE_THRESHOLD = 0.5

# ==========================================
# 1. LOAD THE GUARDRAIL
# ==========================================
print(f"🔄 Loading Guardrail Model from {GUARD_MODEL_PATH}...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(GUARD_MODEL_PATH, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(GUARD_MODEL_PATH).to(device)
    print(f"✅ Guardrail loaded on {device.upper()}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Did you run the training script? Make sure the folder exists.")
    exit()

# Labels Mapping (Must match your training script!)
ID2LABEL = {
    0: "Benign", 
    1: "Harmful Content", 
    2: "Prompt Injection"
}

# ==========================================
# 2. DEFINE FUNCTIONS
# ==========================================

def scan_prompt(text):
    """Passes text through the fine-tuned Guardrail."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    # Get winner
    pred_idx = torch.argmax(probs).item()
    confidence = probs[0][pred_idx].item()
    
    return pred_idx, confidence, probs[0]

def call_ollama(prompt):
    """Sends the safe prompt to the local LLM."""
    payload = {
        "model": TARGET_LLM,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        duration = time.time() - start_time
        return result.get("response", ""), duration
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}", 0

# ==========================================
# 3. THE PIPELINE LOOP
# ==========================================
def main():
    print("\n🛡️  SECURE AI PIPELINE ACTIVE")
    print("-----------------------------------")
    print(f"Target LLM: {TARGET_LLM}")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input(">> You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        # --- STEP 1: GUARDRAIL CHECK ---
        print("\n   [Scanning prompt...]", end="\r")
        label_id, conf, all_probs = scan_prompt(user_input)
        label_name = ID2LABEL[label_id]
        
        # Display Analysis
        print(f"   🔍 Analysis: {label_name} ({conf:.1%})                    ")
        
        # --- STEP 2: DECISION LOGIC ---
        if label_id == 0:
            # BENIGN -> PASS
            print("   ✅ Status: ALLOWED. Sending to LLM...")
            llm_response, duration = call_ollama(user_input)
            print(f"   🤖 {TARGET_LLM}: {llm_response.strip()}")
            print(f"   ⏱️  Response time: {duration:.2f}s\n")
            
        elif label_id == 1:
            # HARMFUL -> BLOCK
            print("   ⚠️  Status: BLOCKED (Safety Violation)")
            print("   🚫 Response: I cannot fulfill this request as it violates safety policies.\n")
            
        elif label_id == 2:
            # INJECTION -> BLOCK & ALERT
            print("   🚨 Status: BLOCKED (Security Alert)")
            print("   🛡️  Response: Potential Prompt Injection detected. This event has been logged.\n")

if __name__ == "__main__":
    main()
