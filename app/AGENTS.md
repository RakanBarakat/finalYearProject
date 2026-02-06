# AGENTS.md

This document provides guidelines for agentic coding agents working in this repository.

## Project Overview

This is a full-stack guardrail system with:
- **Frontend**: Next.js 16 with TypeScript, React 19, Tailwind CSS
- **Backend**: FastAPI with Python, PyTorch, Hugging Face Transformers

## Build Commands

### Frontend (Next.js)

```bash
cd frontend
npm run dev          # Start development server (port 3000)
npm run build        # Production build
npm run start        # Start production server
npm run lint         # Run ESLint
npm run lint -- --fix  # Auto-fix linting issues
```

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000  # Dev server
uvicorn main:app --host 0.0.0.0 --port 8000  # Production
```

## Code Style Guidelines

### TypeScript/React (Frontend)

**Imports:**
- Use absolute imports with `@/` prefix
- Group imports: React → external libraries → internal
- Add `"use client"` at top of client components

**Types:**
```typescript
type Guard = {
  id: number;
  name: string;
  enable_heuristics: boolean;
  enable_classifier: boolean;
  enable_llm: boolean;
  layer_order: string;
};
```

**Naming:**
- Components: PascalCase (`Dashboard`, `GuardBuilder`)
- Hooks: `use` prefix (`useGuards`, `useAuth`)
- Variables/functions: camelCase
- Constants: UPPER_SNAKE_CASE

**Formatting:**
- 2 spaces indentation, use semicolons
- Arrow functions for components/callbacks
- Functional components with hooks

**Error Handling:**
```typescript
try {
  await api.delete(`/guards/${id}`);
  setGuards(guards.filter(g => g.id !== id));
} catch (error) {
  console.error("Failed to delete guardrail:", error);
}
```

### Python (Backend)

**Imports:**
- Standard library → third-party → local application

**Types (Pydantic):**
```python
class GuardCreate(BaseModel):
    name: str
    enable_heuristics: bool = False
    enable_classifier: bool = True
    layer_order: str = "heuristics,classifier,llm"
```

**Naming:**
- Functions/variables: snake_case (`get_db`, `db_path`)
- Constants: UPPER_SNAKE_CASE (`MODELS_DIR`, `DB_PATH`)
- Classes: PascalCase (`GuardRequest`)

**Error Handling:**
```python
if not guard:
    raise HTTPException(status_code=404, detail="Guard not found")
```

**Formatting:**
- PEP 8, 4 spaces, type hints required

## Project Structure

```
app/
├── frontend/           # Next.js 16, TypeScript, Tailwind
│   ├── app/           # App router pages
│   ├── package.json
│   ├── tsconfig.json
│   └── eslint.config.mjs
├── backend/           # FastAPI, PyTorch
│   ├── main.py
│   ├── requirements.txt
│   └── saved_models/
└── AGENTS.md
```

## API Conventions

- Backend: `http://localhost:8000`, CORS for `http://localhost:3000`
- Frontend: axios with base URL `http://localhost:8000`

## Database

- SQLite at `backend/guardrails.db`
- Uses `sqlite3.Row` for dict-like access
- Table: `guards` with guardrail configuration

## Key Dependencies

**Frontend:** next 16.1.3, react 19.2.3, typescript ^5, eslint ^9, tailwindcss ^4, axios, framer-motion, lucide-react

**Backend:** fastapi, uvicorn, torch, transformers, pydantic, python-multipart

## Testing

No test framework configured. Add with:
- Frontend: Vitest/Jest + React Testing Library
- Backend: pytest + FastAPI TestClient
