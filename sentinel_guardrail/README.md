# 🛡️ Sentinel Guardrail Architect

An AI Guardrail Manager for testing and evaluating security guardrails against various attack types.

## Features

- **Layer 1: Text Classifier** - Uses trained transformer models for prompt classification
- **Layer 2: LLM Verifier** - Optional verification using local Ollama models
- **Interactive Mode** - Real-time prompt testing
- **Evaluation Mode** - Batch testing with metrics and visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py --server.maxUploadSize 2000
```

## Configuration

1. **Select Model**: Choose from available model directories in your workspace
2. **Set Threshold**: Adjust confidence threshold for classification
3. **Enable LLM Verification**: Optionally add a second layer using Ollama models
4. **Save Configuration**: Store guardrail settings to database

## Modes

### Interactive Mode
- Test individual prompts in real-time
- Get instant classification results
- Optional third-party verification

### Evaluation Mode
- Upload test datasets (CSV format)
- Run comprehensive evaluations
- View performance metrics and charts
- Identify weaknesses in the guardrail

## Requirements

- Python 3.8+
- Ollama (optional, for LLM verification layer)
- Trained model weights in a local directory
