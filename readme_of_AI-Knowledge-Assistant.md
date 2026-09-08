# AI Knowledge Assistant

> A local-first Retrieval-Augmented Generation (RAG) application for querying uploaded knowledge sources through a grounded conversational interface.

## Overview

AI Knowledge Assistant is an end-to-end RAG application that allows authenticated users to upload documents, index their contents, and ask natural-language questions over the resulting knowledge base.

The system combines document ingestion, local embeddings, vector retrieval, semantic query routing, LLM generation, persistent chat sessions, authentication, access control, and a React-based interface.

The project is intentionally built as a **local-first AI engineering system** rather than a simple LLM chatbot.

### What the system does

```text
Documents
    ↓
Text extraction
    ↓
Chunking
    ↓
Local embeddings
    ↓
ChromaDB
    ↓
User question
    ↓
Semantic query routing
    ↓
Relevant-context retrieval
    ↓
Grounded prompt construction
    ↓
LLM
    ↓
Answer + persisted chat history
```

---

## Key Features

### Document ingestion

Upload supported files and process them into searchable knowledge chunks.

The ingestion pipeline performs:

1. File validation and storage
2. File-type-specific text extraction
3. Page-aware processing where applicable
4. Recursive text chunking
5. Batched embedding generation
6. Vector upsert into ChromaDB
7. Document metadata persistence in PostgreSQL

### Retrieval-Augmented Generation

The RAG pipeline:

1. Authenticates the request and validates document access
2. Classifies the query using semantic intent routing
3. Retrieves relevant chunks from ChromaDB
4. Groups retrieved content by source/page
5. Builds a source-aware context
6. Incorporates relevant conversation history
7. Generates an answer using the configured LLM
8. Persists the conversation turn

The prompt explicitly constrains the model to answer from the supplied context, helping reduce unsupported responses.

### Local embedding models

Embeddings are generated locally with SentenceTransformers.

Current user-facing registry entries:

- `bge-small`
- `qwen-0.6b`
- `qwen3-embedding-0.6b`
- `nomic-v1.5`

The default configured embedding model is `nomic-v1.5`.

> `qwen3-embedding-0.6b-gguf` is present in the codebase but is not currently treated as a user-facing runnable path in the SentenceTransformer implementation.

### LLM provider abstraction

The LLM layer is designed around a provider abstraction with support for:

- Ollama
- Gemini
- OpenAI-compatible APIs

**Ollama is the primary/default runtime path.**

Gemini and OpenAI-compatible providers require their respective credentials and external service availability.

### Conversational sessions

Chat sessions and messages are persisted so users can continue multi-turn conversations instead of treating every request as independent.

### Authentication and access control

The application includes:

- User registration and login
- JWT-based authentication
- Password hashing
- Protected routes
- User-level document/session filtering
- Administrator-only endpoints

### Administration

The application includes an administrative interface for:

- User management
- Application settings
- Model configuration
- Diagnostics and health information

---

## Architecture

The backend follows a layered service-oriented structure:

```text
┌─────────────────────────────────────────────────────┐
│                     React UI                        │
│              React + Vite + Axios                  │
└──────────────────────────┬──────────────────────────┘
                           │ REST API
                           ▼
┌─────────────────────────────────────────────────────┐
│                    FastAPI API                      │
│ Auth │ Documents │ RAG │ Chat │ LLM │ Admin │ Health│
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                  Service Layer                      │
│ Auth │ Document │ Retrieval │ RAG │ LLM │ Settings │
└──────────────┬────────────────────────┬─────────────┘
               │                        │
               ▼                        ▼
      ┌────────────────┐       ┌─────────────────────┐
      │   PostgreSQL   │       │      ChromaDB       │
      │ Auth + metadata│       │ Vector persistence  │
      └────────────────┘       └──────────┬──────────┘
                                         ▲
                                         │
                                ┌────────┴────────┐
                                │ Embedding Service│
                                │ SentenceTransformers
                                └─────────────────┘

                        LLM Provider Layer
                   ┌─────────┬──────────┬──────────┐
                   │ Ollama  │  Gemini  │ OpenAI-* │
                   └─────────┴──────────┴──────────┘
```

The repository separates API routers, services, repositories, domain models, and external systems.

This structure is intended to keep transport logic, business logic, persistence, and AI-provider integrations independently maintainable.

---

## End-to-End Workflows

### 1. Document ingestion

```text
Upload
  ↓
Extension validation
  ↓
Local file storage
  ↓
Document metadata record
  ↓
Type-specific extraction
  ↓
Recursive chunking
  ↓
Batch embedding generation
  ↓
ChromaDB vector upsert
  ↓
Document metadata update
```

### 2. Question answering

```text
User question
     ↓
JWT authentication
     ↓
Document/session access validation
     ↓
Semantic query intent routing
     ↓
Vector retrieval from ChromaDB
     ↓
Page/source-aware context assembly
     ↓
Conversation history integration
     ↓
Grounded prompt
     ↓
LLM provider
     ↓
Answer
     ↓
Chat history persistence
```

---

## Supported Upload Types

The current upload handler supports:

<details>
<summary>View exact supported extensions</summary>

### Documents

`.pdf` `.doc` `.docx` `.ppt` `.pptx` `.xls` `.xlsx`

### Text / data

`.txt` `.md` `.csv` `.log`

### Source code

`.py` `.js` `.ts` `.tsx` `.jsx` `.java` `.c` `.cc` `.cpp` `.cxx`
`.h` `.hpp` `.rs` `.go` `.kt` `.swift` `.php` `.rb` `.cs` `.scala`
`.sh` `.bash`

### Structured / web / configuration

`.json` `.html` `.htm` `.css` `.sql` `.xml` `.yaml` `.yml`

</details>

---

## Technology Stack

| Area | Technologies |
|---|---|
| Backend | Python, FastAPI, SQLAlchemy, Pydantic Settings |
| Database | PostgreSQL |
| Vector store | ChromaDB |
| Embeddings | SentenceTransformers |
| LLM runtime | Ollama |
| Optional providers | Gemini, OpenAI-compatible APIs |
| Frontend | React 19, Vite |
| HTTP client | Axios |
| Auth | JWT, bcrypt |
| Testing | pytest, HTTPX |
| Code quality | Ruff |
| Local infrastructure | Docker Compose |

---

## Project Structure

```text
backend/
├── app/
│   ├── routers/
│   │   ├── auth.py
│   │   ├── document.py
│   │   ├── rag.py
│   │   ├── chat.py
│   │   ├── health.py
│   │   ├── admin.py
│   │   ├── admin_llm.py
│   │   └── llm.py
│   │
│   ├── services/
│   │   ├── vector/
│   │   ├── document/
│   │   ├── rag/
│   │   ├── llm/
│   │   ├── auth/
│   │   └── ...
│   │
│   ├── repositories/
│   ├── models/
│   └── ...
│
├── pyproject.toml
└── ...

frontend/
├── src/
│   ├── components/
│   ├── context/
│   ├── pages/
│   ├── api.js
│   ├── App.jsx
│   └── main.jsx
├── package.json
└── ...
```

---

## API Surface

### Authentication

```http
POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/auth/me
```

### Documents

```http
POST   /api/v1/documents/upload
GET    /api/v1/documents
GET    /api/v1/documents/{id}
DELETE /api/v1/documents/{id}
```

### RAG and chat

```http
POST   /api/v1/rag/ask
POST   /api/v1/chat/sessions
GET    /api/v1/chat/sessions
GET    /api/v1/chat/sessions/{session_id}
POST   /api/v1/chat/sessions/{session_id}/messages
DELETE /api/v1/chat/sessions/{session_id}
```

### Health and model discovery

```http
GET /api/v1/health
GET /api/v1/health/live
GET /api/v1/llm/models
GET /api/v1/llm/models/{model_id}/status
```

### Administration

```http
GET    /api/v1/admin/stats
GET    /api/v1/admin/users
POST   /api/v1/admin/users
PATCH  /api/v1/admin/users/{id}
DELETE /api/v1/admin/users/{id}
GET    /api/v1/admin/settings
PATCH  /api/v1/admin/settings
GET    /api/v1/admin/diagnostics
```

---

## Local Setup

### Prerequisites

Install or have available:

- Python environment managed with `uv`
- Node.js / npm
- Docker Desktop or Docker Engine
- Ollama
- The embedding model files expected by the repository
- A configured `.env` file

### 1. Start PostgreSQL

From the repository root:

```bash
docker compose up -d postgres
```

### 2. Start the backend

```bash
uv sync --directory backend
uv run --directory backend uvicorn app.main:app --host 127.0.0.1 --port 8500 --reload --reload-dir app
```

Backend:

```text
http://127.0.0.1:8500
```

### 3. Start the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Use the Vite URL shown in the terminal to open the frontend.

### 4. Ollama

The default configuration uses Ollama as the primary LLM provider.

The configured default model is:

```text
qwen3:8b
```

Make sure Ollama is installed, running, and the required model is available before using the chat workflow.

> Exact environment variables and local model-directory expectations should be documented in `.env.example` and the repository configuration before this README is treated as the final setup guide.

---

## Configuration

Configuration is centralized in the backend settings layer and loaded from environment variables / `.env`.

The application includes settings for:

- PostgreSQL connection
- Embedding model selection
- LLM provider/model selection
- Ollama base URL
- Gemini credentials
- JWT secret, algorithm and expiration
- RAG defaults
- Chat TTL/history behaviour

### Security note

Do not commit `.env` files, API credentials, model secrets, or production credentials to the repository.

---

## Testing

The repository includes a pytest-based test suite intended primarily for **component-level and integration-oriented validation** of application behaviour.

The tests are not intended to benchmark the assistant against external commercial AI systems.

### Current testing status

The core application components have been functionally exercised and the application runs end-to-end.

Some older tests require maintenance because the implementation evolved after those tests were written. These tests should be updated to match the current interfaces and contracts before claiming a fully green suite.

Therefore, this repository does **not** currently claim a complete passing test suite or a formal code-coverage percentage.

---

## Engineering Decisions

### Local-first architecture

The application keeps the primary AI workflow local where practical, using local embeddings, Ollama and persistent local vector storage.

This makes the project easier to reproduce and inspect during development and portfolio demonstration.

### PostgreSQL + ChromaDB

PostgreSQL handles structured application state such as users, sessions and document metadata.

ChromaDB handles vector storage and similarity retrieval for document chunks.

### Provider abstraction

LLM integrations are isolated behind provider-specific implementations so the RAG layer does not have to be tightly coupled to one external API.

### Layered backend

The router → service → repository separation makes it easier to test components independently and evolve the application without putting business logic directly into API handlers.

### Grounded generation

The RAG prompt is constructed around retrieved source context and explicitly instructs the model to avoid unsupported information.

---

## Screenshots & Demo

### RAG Workflow

The primary workflow demonstrates document-grounded conversational
question answering after a knowledge source has been uploaded.

![RAG chat](docs/screenshots/rag-chat.png)

### Application Interface

![Main dashboard](docs/screenshots/dashboard.png)

### Document Management

![Document upload](docs/screenshots/document-upload.png)

### Authentication

![Login screen](docs/screenshots/login.png)

### Administration

![Admin dashboard](docs/screenshots/admin-dashboard.png)

> **Repository note:** The screenshots above are stored under
> `docs/screenshots/` and referenced using repository-relative Markdown
> paths so they render directly on GitHub.

---

## Known Limitations

This repository is a **working local-first RAG prototype with production-oriented architectural patterns**, not a hardened enterprise deployment platform.

Current limitations include:

- Document ingestion is synchronous
- Files are stored on the local filesystem
- ChromaDB is locally persistent
- Ollama is the primary runtime path
- Production observability is limited
- Full application containerization is not yet provided
- Some administrative settings are not fully connected to runtime behaviour
- Security controls still require production hardening
- The current test suite contains tests that need maintenance after implementation changes

---

## Production Hardening Roadmap

Potential next-stage improvements include:

- Background workers for document ingestion
- Asynchronous processing and progress tracking
- Hybrid keyword + semantic retrieval
- Retrieval reranking
- Retrieval quality and groundedness evaluation
- Citation quality checks
- Prompt versioning
- Structured logging and distributed tracing
- Metrics, dashboards and alerting
- Rate limiting and abuse protection
- Secure secret management
- Full-stack containerization
- Model fallback/routing policies
- Stronger file validation and content scanning
- CI/CD and deployment automation

---

## Project Status

**Functional prototype / portfolio project**

The core document ingestion, vector retrieval, RAG, authentication, chat, administration and frontend workflows are implemented.

The primary remaining work is production hardening, evaluation, operational tooling, and maintenance of tests that no longer match newer implementation changes.

---

## Author

**Pritam Sunil Mahajan**

Master's in Artificial Intelligence — IIT Ropar

[LinkedIn](YOUR_LINKEDIN_URL)

---

## License

> **TODO:** Add the project's actual license here.

