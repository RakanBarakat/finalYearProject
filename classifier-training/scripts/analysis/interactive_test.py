import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 80)
print("INTERACTIVE PROMPT INJECTION DETECTOR")
print("=" * 80)

# Load models
print("\nLoading models...")
try:
    binary_tokenizer = AutoTokenizer.from_pretrained("./stage1_model")
    binary_model = AutoModelForSequenceClassification.from_pretrained("./stage1_model")
    print("✓ Stage 1 model loaded")
except Exception as e:
    print(f"✗ Stage 1 model failed to load: {e}")
    exit(1)

try:
    multiclass_tokenizer = AutoTokenizer.from_pretrained("./stage2_model")
    multiclass_model = AutoModelForSequenceClassification.from_pretrained("./stage2_model")
    print("✓ Stage 2 model loaded")
except Exception as e:
    print(f"✗ Stage 2 model failed to load: {e}")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
binary_model.to(device)
multiclass_model.to(device)
print(f"✓ Models moved to: {device}")

# Get label mappings
binary_id2label = binary_model.config.id2label
multiclass_id2label = multiclass_model.config.id2label

print(f"\nBinary labels: {list(binary_id2label.values())}")
print(f"Multi-class labels: {list(multiclass_id2label.values())}")


def classify_prompt(text):
    """
    Two-stage classification:
    1. Stage 1: Benign / Harmful Content / Attack
    2. Stage 2: Attack Type (Semantic / Obfuscation / Vector) - only if Attack
    """
    print(f"\n{'=' * 80}")
    print(f"PROMPT: {text}")
    print('=' * 80)
    
    # Stage 1: 3-way classification
    inputs = binary_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = binary_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        stage1_pred_id = torch.argmax(probs, dim=-1).item()
        stage1_label = binary_id2label[stage1_pred_id]
        stage1_confidence = probs[0][stage1_pred_id].item()
    
    print(f"\n🔍 Stage 1: Intent Classification")
    print(f"   Result: {stage1_label}")
    print(f"   Confidence: {stage1_confidence:.1%}")
    print(f"   ", end="")
    for i, label in binary_id2label.items():
        print(f"{label}: {probs[0][i].item():.1%}", end=" | " if i < len(binary_id2label)-1 else "\n")
    
    # Stage 2: Attack type classification (only if Attack)
    if stage1_label == "Attack":
        inputs = multiclass_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = multiclass_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get actual prediction
            pred_id = torch.argmax(probs, dim=-1).item()
            actual_prediction = multiclass_id2label[pred_id]
            actual_confidence = probs[0][pred_id].item()
            
            # Filter out "Non-Attack" to show attack categories
            attack_probs = []
            non_attack_conf = None
            for i, label in multiclass_id2label.items():
                if label == "Non-Attack":
                    non_attack_conf = probs[0][i].item()
                else:
                    attack_probs.append((label, probs[0][i].item()))
            
            # Sort by confidence
            attack_probs.sort(key=lambda x: x[1], reverse=True)
            top_label, top_confidence = attack_probs[0]
            
            print(f"\n🎯 Stage 2: Attack Type Classification")
            print(f"   Attack Type: {top_label}")
            print(f"   Confidence: {top_confidence:.1%}")
            
            # Check for model conflicts
            if actual_prediction == "Non-Attack" and actual_confidence > 0.50:
                print(f"\n   ⚠️  MODEL CONFLICT: Stage 1 says Attack, but Stage 2 says Non-Attack")
                print(f"   ⚠️  This prompt may need to be added to training data")
            elif top_confidence < 0.50:
                print(f"\n   ⚠️  LOW CONFIDENCE: Attack detected but type is uncertain")
                print(f"   ⚠️  This prompt may need to be added to training data")
    else:
        reason = "Benign" if stage1_label == "Benign" else "Harmful Content"
        print(f"\n✅ Stage 2: Skipped ({reason} detected)")
    
    print('=' * 80)


print("\n" + "=" * 80)
print("INTERACTIVE MODE")
print("=" * 80)
print("Enter prompts to test (type 'quit' or 'exit' to stop):\n")

while True:
    try:
        user_input = input("\n>>> ")
        
        if user_input.lower().strip() in ['quit', 'exit', 'q', '']:
            break
        
        if not user_input.strip():
            continue
        
        classify_prompt(user_input)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")

print("\n\n👋 Goodbye!")
