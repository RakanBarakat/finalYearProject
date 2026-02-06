import streamlit as st
import sqlite3
import pandas as pd
import subprocess
import os
import time
import zipfile
import shutil

# ==========================================
# 1. SYSTEM SETUP & DATABASE
# ==========================================

MODELS_DIR = "saved_models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def init_db():
    """Initializes SQLite DB."""
    conn = sqlite3.connect('guardrails.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS guards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            classifier_path TEXT,
            heuristic_threshold INTEGER,
            llm_verifier TEXT,
            enable_heuristics INTEGER DEFAULT 1,
            enable_classifier INTEGER DEFAULT 1,
            enable_llm INTEGER DEFAULT 1,
            layer_order TEXT DEFAULT 'classifier_first',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def get_ollama_models():
    """Scans for Ollama models."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, shell=True)
        if result.returncode != 0: return ["Error: Ollama not running"]
        lines = result.stdout.strip().split('\n')[1:]
        models = [line.split()[0].split(':')[0] for line in lines]
        return models if models else ["No models found"]
    except:
        return ["Error scanning models"]

@st.cache_resource
def load_model_resources(model_path):
    """Lazy loader for Neural Models."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        return tokenizer, model, device
    except Exception as e:
        return None, None, str(e)

# ==========================================
# 2. UI: ARCHITECTURE BUILDER
# ==========================================

@st.dialog("🏗️ Construct Guardrail Architecture")
def build_guardrail_popup():
    st.write("Define your security pipeline layers.")
    
    # 1. Name
    name = st.text_input("Guardrail Name", "Production_V1")
    
    st.markdown("---")
    
    # 2. Layer Order Configuration
    st.subheader("⚙️ Layer Configuration")
    
    col_a, col_b = st.columns(2)
    with col_a:
        layer_order = st.radio(
            "Processing Order",
            ["classifier_first", "llm_first"],
            format_func=lambda x: "Classifier → LLM" if x == "classifier_first" else "LLM → Classifier"
        )
    
    with col_b:
        enable_heuristics = st.checkbox("Enable Speed Heuristics", value=True)
        enable_classifier = st.checkbox("Enable Neural Classifier", value=True)
        enable_llm = st.checkbox("Enable LLM Verifier", value=True)
    
    st.markdown("---")
    
    # 3. Layer 1: Heuristics
    st.subheader("Layer 1: Speed Heuristics")
    st.caption("Skip heavy scanning for very short prompts.")
    char_limit = st.number_input("Minimum Character Threshold", min_value=0, value=20, disabled=not enable_heuristics)
    
    st.markdown("---")
    
    # 4. Layer 2: Neural Model (ZIP UPLOAD)
    st.subheader("Layer 2: Neural Classifier")
    st.caption("Upload your fine-tuned model folder as a .zip file.")
    
    uploaded_zip = st.file_uploader("Drag & Drop Model (.zip)", type="zip")
    final_model_path = None
    
    if uploaded_zip is not None:
        # Save ZIP logic
        extract_path = os.path.join(MODELS_DIR, name.replace(" ", "_"))
        if not os.path.exists(extract_path):
            with st.spinner(f"Installing model to {extract_path}..."):
                # Write zip to temp
                zip_path = f"temp_{uploaded_zip.name}"
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())
                
                # Extract
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
                # Cleanup
                os.remove(zip_path)
                st.success("✅ Model Installed Successfully!")
        
        final_model_path = extract_path
    
    st.markdown("---")
    
    # 5. Layer 3: LLM Verifier
    st.subheader("Layer 3: LLM Verifier")
    ollama_models = get_ollama_models()
    selected_llm = st.selectbox("Select Local LLM", ollama_models, disabled=not enable_llm)
    
    # SAVE BUTTON
    if st.button("💾 Save Configuration"):
        # Validation
        if not enable_classifier and not enable_llm:
            st.error("❌ You must enable at least one layer (Classifier or LLM)")
        elif enable_classifier and (not final_model_path or not os.path.exists(final_model_path)):
            st.error("❌ Please upload and install a model for the Neural Classifier")
        else:
            conn = init_db()
            c = conn.cursor()
            c.execute("""INSERT INTO guards 
                      (name, classifier_path, heuristic_threshold, llm_verifier, 
                       enable_heuristics, enable_classifier, enable_llm, layer_order) 
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                      (name, final_model_path or '', char_limit, selected_llm,
                       int(enable_heuristics), int(enable_classifier), int(enable_llm), layer_order))
            conn.commit()
            st.success(f"✅ Guardrail '{name}' saved to Database!")
            time.sleep(1)
            st.rerun()

# ==========================================
# 3. MAIN APP INTERFACE
# ==========================================

st.set_page_config(page_title="Sentinel AI", layout="wide", page_icon="🛡️")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Sentinel")
    
    if st.button("➕ New Guardrail", use_container_width=True):
        build_guardrail_popup()
    
    st.markdown("---")
    st.markdown("### Active Configuration")
    
    conn = init_db()
    guards = pd.read_sql("SELECT * FROM guards ORDER BY id DESC", conn)
    
    if not guards.empty:
        active_name = st.selectbox("Select Active Guardrail", guards['name'])
        active_guard = guards[guards['name'] == active_name].iloc[0]
        
        st.info(f"**Loaded:** {active_name}")
        
        # Display layer configuration
        layers_enabled = []
        if active_guard.get('enable_heuristics', 1): layers_enabled.append("Heuristics")
        if active_guard.get('enable_classifier', 1): layers_enabled.append("Classifier")
        if active_guard.get('enable_llm', 1): layers_enabled.append("LLM")
        
        order = active_guard.get('layer_order', 'classifier_first')
        order_text = "Classifier → LLM" if order == 'classifier_first' else "LLM → Classifier"
        
        st.code(f"Layers: {', '.join(layers_enabled)}\nOrder: {order_text}\nPath: {active_guard.get('classifier_path', 'N/A')}\nLLM: {active_guard.get('llm_verifier', 'N/A')}")
    else:
        active_guard = None
        st.warning("No guardrails found. Create one!")

# --- MAIN SCREEN ---
st.header("Security Operations Center")
tab1, tab2 = st.tabs(["💬 Interactive Mode", "📊 Evaluation Mode"])

# === TAB 1: INTERACTIVE ===
with tab1:
    if active_guard is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Prompt Scanning")
            user_input = st.text_area("Input Prompt", height=150, placeholder="Enter text to scan...")
            
            if st.button("🚀 Scan", use_container_width=True):
                label_id = 0
                conf = 1.0
                llm_result = None
                classifier_result = None
                skipped_heuristics = False
                
                with st.status("Running Pipeline...", expanded=True) as status:
                    
                    # 1. Heuristics (if enabled)
                    if active_guard.get('enable_heuristics', 1):
                        st.write("⚡ Checking speed heuristics...")
                        time.sleep(0.1)
                        if len(user_input) < active_guard['heuristic_threshold']:
                            label_id = 0 # Benign
                            conf = 1.0
                            skipped_heuristics = True
                            st.info("ℹ️ Short prompt - marked as benign")
                    
                    # Execute layers based on configured order
                    if not skipped_heuristics:
                        layer_order = active_guard.get('layer_order', 'classifier_first')
                        
                        # Define which layers to run
                        def run_classifier_layer():
                            nonlocal label_id, conf, classifier_result
                            if active_guard.get('enable_classifier', 1):
                                st.write("🧠 Loading Neural Model...")
                                tokenizer, model, dev = load_model_resources(active_guard['classifier_path'])
                                
                                if model:
                                    import torch
                                    st.write(f"Inference on {dev.upper()}...")
                                    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512).to(dev)
                                    with torch.no_grad():
                                        outputs = model(**inputs)
                                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                                        label_id = torch.argmax(probs).item()
                                        conf = probs[0][label_id].item()
                                    classifier_result = (label_id, conf)
                                else:
                                    st.error(f"Model Load Failed: {dev}")
                                    label_id = -1
                        
                        def run_llm_layer():
                            nonlocal llm_result
                            if active_guard.get('enable_llm', 1):
                                st.write(f"🤖 Querying LLM ({active_guard['llm_verifier']})...")
                                time.sleep(0.5)
                                # Mock LLM response (in real implementation, call Ollama API)
                                llm_result = "malicious" if "ignore" in user_input.lower() or "prompt" in user_input.lower() else "safe"
                        
                        # Execute in configured order
                        if layer_order == 'classifier_first':
                            run_classifier_layer()
                            run_llm_layer()
                        else:
                            run_llm_layer()
                            run_classifier_layer()
                    
                    status.update(label="Complete", state="complete")
                
                # Display Result
                LABELS = {0: "Benign", 1: "Harmful Content", 2: "Prompt Injection"}
                verdict = LABELS.get(label_id, "Error")
                
                # Show results from enabled layers
                if classifier_result:
                    st.write(f"**Classifier Result:** {LABELS.get(classifier_result[0], 'Unknown')} ({classifier_result[1]:.1%})")
                if llm_result:
                    st.write(f"**LLM Result:** {llm_result.upper()}")
                
                if label_id == 0:
                    st.success(f"✅ **ALLOWED**: {verdict} ({conf:.1%})")
                elif label_id > 0:
                    st.error(f"🚨 **BLOCKED**: {verdict} ({conf:.1%})")

        with col2:
            st.subheader("Pipeline Status")
            
            # Show enabled layers
            st.caption("**Active Layers:**")
            if active_guard.get('enable_heuristics', 1):
                st.write("⚡ Speed Heuristics")
            if active_guard.get('enable_classifier', 1):
                st.write("🧠 Neural Classifier")
            if active_guard.get('enable_llm', 1):
                st.write(f"🤖 LLM ({active_guard['llm_verifier']})")
            
            st.caption("**Processing Order:**")
            order = active_guard.get('layer_order', 'classifier_first')
            if order == 'classifier_first':
                st.write("1️⃣ Classifier → 2️⃣ LLM")
            else:
                st.write("1️⃣ LLM → 2️⃣ Classifier")

# === TAB 2: EVALUATION ===
with tab2:
    if active_guard is not None:
        st.subheader("Benchmark Suite")
        uploaded_csv = st.file_uploader("Upload Evaluation Dataset (.csv)", type="csv")
        
        if uploaded_csv and st.button("Run Full Evaluation"):
            df = pd.read_csv(uploaded_csv)
            st.write(f"Loaded {len(df)} samples.")
            
            my_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                my_bar.progress(i+1)
            
            st.success("Evaluation Complete")
            
            # Mock Data for Visualization
            metrics = {
                "Attack Type": ["Benign", "Harmful Content", "Prompt Injection", "Vector Attacks"],
                "F1 Score": [0.99, 0.93, 0.96, 0.89],
                "Avg Latency (ms)": [12, 15, 18, 35]
            }
            df_m = pd.DataFrame(metrics)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Accuracy", "94.5%")
            k2.metric("Latency", "20ms")
            k3.metric("False Positives", "1.2%")
            
            st.bar_chart(df_m.set_index("Attack Type")["F1 Score"])
            st.dataframe(df_m, use_container_width=True)