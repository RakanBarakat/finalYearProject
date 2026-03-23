# Sentinel — Modular AI Guardrail Benchmarking Platform

A platform for building, testing, and benchmarking AI guardrail configurations. Users assemble guards from mix-and-match layers (text classifier, LLM verifier, RAG augmentation), then evaluate them against standard datasets with full statistical metrics.

## Architecture

Guards are built from two independent layers, each toggleable:

- **Text Classifier** — Fine-tuned HuggingFace models (DeBERTa, PromptGuard) for binary prompt classification. Fast, runs locally on GPU/CPU.
- **LLM Verifier** — External LLM (NanoGPT cloud or Ollama local) with a security classification prompt. Slower but more nuanced.
- **RAG Augmentation** — FAISS vector index retrieves similar known threats/benign examples and injects them into the LLM prompt for context-aware classification.

### Guard Configurations

| Configuration | How it works |
|---|---|
| Classifier only | Binary classification via HuggingFace model |
| LLM only | LLM classifies as BENIGN / HARMFUL / INJECTION |
| LLM + RAG | Same as above, with retrieved threat examples in prompt |
| Classifier + LLM | Three-zone risk gating — classifier auto-decides safe/dangerous, LLM handles grey zone |
| Classifier + LLM + RAG | Double-layer with RAG-augmented LLM for grey-zone tie-breaks |

### Three-Zone Risk Gating

When both layers are enabled, the classifier's risk score maps to zones:

- **Zone A** (safe): score < lower bound — auto-allow, skip LLM
- **Zone C** (grey): between bounds — sent to LLM for tie-break
- **Zone B** (dangerous): score > upper bound — auto-block, skip LLM

This reduces expensive LLM calls by only invoking them for ambiguous cases. Bounds are configurable per guard via UI sliders.

## Tech Stack

| Component | Stack |
|---|---|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4, Framer Motion, Recharts |
| Backend | FastAPI, Python, PyTorch (CUDA 12.1), Transformers, FAISS, Sentence-Transformers |
| Database | SQLite |
| LLM Providers | NanoGPT (cloud), Ollama (local) |

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) CUDA-compatible GPU for faster classifier inference
- (Optional) Ollama installed locally for local LLM support

### Backend

```bash
cd app/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd app/frontend
npm install
npm run dev
```

### RAG Index (one-time, required for RAG features)

```bash
cd app/backend
python build_rag_index.py
```

## Evaluation

The platform benchmarks guards against 4 curated datasets:

| Dataset | Samples | Type | Focus |
|---|---|---|---|
| NotInject (HuggingFace) | 339 | Benign-only | False positive rate |
| StrongREJECT (HuggingFace) | 313 | Malicious-only | Attack detection rate |
| Prompt Injection Eval | 500 | Mixed | Injection, harmful, obfuscation |
| Vector & Obfuscation Eval | 500 | Mixed | Indirect vector attacks, unicode |

### Metrics

- **Accuracy** — overall correctness
- **Precision** — % of blocked content that was actually malicious
- **Recall** — % of attacks successfully detected
- **F1 Score** — harmonic mean of precision and recall
- **FPR** — false positive rate (benign content incorrectly blocked)

Results include confusion matrices, per-attack-type breakdowns, zone distribution analytics, and latency timing. Exportable as plain text, Markdown, or LaTeX tables.

## Classifier Training

The `classifier-training/` directory contains the full training pipeline for fine-tuning DeBERTa and PromptGuard models:

- `scripts/training/` — fine-tuning scripts
- `scripts/augmentation/` — data augmentation
- `scripts/analysis/` — model analysis and testing
- `scripts/utilities/` — dataset generation and utilities
- `data/training/` — training datasets
- `data/evaluation/` — evaluation datasets

## Project Structure

```
app/
  backend/
    main.py              # FastAPI application (~3700 lines)
    rag_engine.py         # FAISS RAG retrieval engine
    build_rag_index.py    # One-time RAG index builder
    requirements.txt
  frontend/
    app/page.tsx          # React single-page app (~4100 lines)
    app/globals.css
    package.json
classifier-training/
  scripts/               # Training, augmentation, analysis scripts
  data/                  # Training and evaluation datasets
  models/                # Saved model checkpoints (gitignored)
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/scan` | Classify a single prompt against a guard |
| POST | `/evaluate` | Benchmark a guard on a dataset |
| POST | `/compare` | Compare two guards on the same dataset |
| POST | `/evaluate-all` | Benchmark across all 4 datasets |
| POST | `/compare-all` | Compare two guards across all 4 datasets |
| POST | `/save-guard` | Create a new guard |
| GET | `/guards` | List all guards |
| GET | `/docs` | Swagger API documentation |
