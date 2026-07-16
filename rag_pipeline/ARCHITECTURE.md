# System Architecture

**Version:** 1.3  
**Last Updated:** July 2026

This document describes the complete architecture of the RAG Pipeline system, including data flow, component interactions, and infrastructure design.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Data Flow](#data-flow)
3. [Component Architecture](#component-architecture)
4. [Docker Architecture](#docker-architecture)
5. [Storage Architecture](#storage-architecture)
6. [Performance Characteristics](#performance-characteristics)

---

## High-Level Overview

### System Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE SYSTEM                          │
│                                                                      │
│  INPUT: Scikit-learn HTML Docs (420 files)                         │
│         ↓                                                           │
│  [1] PREPROCESSING → Structured JSON (416 files, ~4 MB)           │
│         ↓                                                           │
│  [2] CHUNKING → 3 Strategies → ~4,000 Total Chunks                │
│         ├─ Fixed: ~1,600 chunks (512 tokens, 50 overlap)          │
│         ├─ Semantic: ~740 chunks (natural boundaries)              │
│         └─ Hierarchical: ~1,700 chunks (structure-aware)           │
│         ↓                                                           │
│  [3] EMBEDDING → Sentence-Transformers (all-MiniLM-L6-v2)         │
│         ↓                                                           │
│  [4] VECTOR STORE → ChromaDB (3 collections, ~86 MB)              │
│      BM25 INDEX → Keyword search indices (per strategy)           │
│         ↓                                                           │
│  [5] QUERY PROCESSING                                              │
│         ├─ Query Rewriting (LLM-based enhancement)                │
│         ├─ Hybrid Search (BM25 + semantic with RRF)               │
│         └─ Cross-Encoder Reranking (ms-marco-MiniLM)              │
│         ↓                                                           │
│  [6] GENERATION (Optional) → Ollama + Llama 3.2 → Answer          │
│                                                                      │
│  OUTPUT: Retrieved chunks OR Natural language answer with citations │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

| Component | Technology | Why Chosen |
|-----------|-----------|------------|
| **Vector DB** | ChromaDB | Local, persistent, good for MVP |
| **Embeddings** | sentence-transformers | Free, local, standard in industry |
| **LLM** | Ollama + Llama 3.2 | Free, local, no API keys needed |
| **Keyword Search** | BM25 (rank-bm25) | Industry standard, complements semantic |
| **Reranking** | Cross-encoder (ms-marco) | High precision, lazy-loaded |
| **Orchestration** | Docker Compose | Multi-container management |
| **Config** | YAML | Human-readable, comments supported |
| **State** | JSON files | Simple, transparent, easy to debug |

---

## Data Flow

### Detailed Pipeline Flow

```
Raw HTML (420 docs)
    ↓ [ITERATION 1: Preprocessing]
Structured JSON (416 files, ~4 MB — 4 of 420 inputs skipped for <50 chars)
    ├─ doc_id, title, content, doc_type
    └─ metadata (sections, file_size, etc.)
    ↓ [ITERATION 2: Chunking]
Text Chunks (~4,000 total)
    ├─ Fixed: ~1,600 chunks (~384 words each)
    ├─ Semantic: ~740 chunks (~1000 words each)
    └─ Hierarchical: ~1,700 chunks (variable size)
    ↓ [ITERATION 2: Embedding]
Vector Embeddings (384 dimensions each)
    ↓ [ITERATION 2: Storage]
ChromaDB Collections (3 separate)
    ├─ Collection "fixed": ~1,600 embeddings
    ├─ Collection "semantic": ~740 embeddings
    └─ Collection "hierarchical": ~1,700 embeddings
    ↓ [ITERATION 3: Query]
User Query → Query Rewriting (LLM) → Rewritten Query
    ↓ [ITERATION 3: Embedding]
Query Embedding (384 dims)
    ↓ [ITERATION 3: Hybrid Search]
    ├─ Semantic Search (ChromaDB, top-50)
    ├─ Keyword Search (BM25, top-50)
    └─ RRF Fusion (alpha=0.7)
    ↓ [ITERATION 3: Reranking]
Cross-Encoder Reranking (ms-marco-MiniLM)
    ↓ [ITERATION 3: Results]
Top-K Relevant Chunks (default: 10)
    ├─ Ranked by rerank score
    └─ Includes hybrid search metadata
    ↓ [ITERATION 4: Generation - Optional]
RAG Prompt (query + context + instructions)
    ↓
Ollama API (Llama 3.2 3B)
    ↓
Natural Language Answer + Inline Citations [1][2]
```

### Data Transformations

**1. HTML → JSON (Preprocessing)**
```
Input:  <html><body><h1>StandardScaler</h1><p>Standardize features...</p></body></html>
Output: {"doc_id": "api__StandardScaler", "content": "StandardScaler\nStandardize...", ...}
```

**2. JSON → Chunks (Chunking)**
```
Input:  Long document text (5000 words)
Output: [Chunk(0, 384 words), Chunk(1, 384 words), ...]  # Fixed strategy example
```

**3. Chunks → Embeddings (Embedding)**
```
Input:  "StandardScaler removes the mean and scales to unit variance"
Output: [0.023, -0.145, 0.892, ...] (384 dimensions)
```

**4. Query → Results (Retrieval)**
```
Input:  "How do I use StandardScaler?"
Step 1: Rewrite query (LLM) → "StandardScaler normalize features preprocessing"
Step 2: Embed query → [0.012, -0.098, ...]
Step 3: Semantic search (ChromaDB) → Top-50 by similarity
Step 4: Keyword search (BM25) → Top-50 by term matching
Step 5: RRF Fusion (alpha=0.7) → Merged ranking
Step 6: Cross-encoder reranking → Reorder by relevance
Output: [{"chunk_id": "...", "rerank_score": 2.34, "content": "..."}]
```

**5. Results → Answer (Generation - Optional)**
```
Input:  Retrieved chunks + query
Step 1: Build prompt with context
Step 2: Send to Ollama
Step 3: Extract citations from response
Output: Natural language answer with [1][2] citations + sources list
```

---

## Component Architecture

### Module Structure

```
src/
├── preprocessing/         # [Iteration 1] HTML → JSON
│   ├── html_parser.py          # BeautifulSoup-based parsing
│   └── corpus_processor.py     # Pipeline orchestration
│
├── chunking/             # [Iteration 2] JSON → Chunks
│   ├── base_chunker.py         # Abstract base class
│   ├── fixed_chunker.py        # Fixed-size chunking
│   ├── semantic_chunker.py     # Semantic boundaries
│   └── hierarchical_chunker.py # Structure-aware chunking
│
├── retrieval/            # [Iterations 2 & 3] Embeddings & Search
│   ├── embedder.py             # Sentence-transformers wrapper
│   ├── vector_store.py         # ChromaDB interface
│   ├── indexer.py              # Build vector indices
│   ├── query_processor.py      # Query orchestration
│   ├── bm25_index.py           # BM25 keyword search index
│   ├── hybrid_searcher.py      # RRF fusion (BM25 + semantic)
│   ├── query_rewriter.py       # LLM-based query rewriting
│   └── reranker.py             # Cross-encoder reranking
│
├── generation/           # [Iteration 4] LLM Integration
│   ├── llm_client.py           # Ollama API client
│   ├── prompt_builder.py       # RAG prompt templates
│   └── answer_generator.py     # Generation pipeline
│
├── evaluation/           # [Iteration 5] Metrics & Analysis
│   ├── evaluator.py            # Main evaluation orchestrator
│   ├── metrics.py              # Recall@k, MRR, NDCG
│   ├── llm_judge.py            # LLM-based quality judging
│   ├── test_loader.py          # Test set loading
│   └── results_analyzer.py     # Statistical analysis
│
└── utils/                # [Iteration 1] Shared Infrastructure
    ├── config.py               # YAML configuration management
    └── logger.py               # Structured logging
```

### Key Interfaces

**BaseChunker (Abstract Base Class)**
```python
class BaseChunker(ABC):
    @abstractmethod
    def chunk_document(self, document: Dict) -> List[Chunk]:
        """Chunk a single document"""
        
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy identifier"""
```

**Embedder (Embedding Generation)**
```python
class Embedder:
    def embed(self, texts: Union[str, List[str]], 
              normalize: bool = True) -> np.ndarray:
        """Generate embeddings for text(s)"""
        
    def get_model_info(self) -> Dict:
        """Get model metadata"""
```

**VectorStore (ChromaDB Wrapper)**
```python
class VectorStore:
    def create_collection(self, name: str) -> None
    def add_documents(self, collection_name: str, chunks: List[Chunk], 
                      embeddings: np.ndarray) -> None
    def query(self, collection_name: str, query_embedding: np.ndarray, 
              top_k: int) -> List[Dict]
```

**QueryProcessor (Retrieval Orchestration)**
```python
class QueryProcessor:
    def process_query(self, query_text: str, strategy: str, 
                      top_k: int) -> Dict[str, Any]:
        """Process query with semantic search"""
        
    def process_query_hybrid(self, query_text: str, strategy: str,
                             top_k: int, alpha: float) -> Dict[str, Any]:
        """Process query with hybrid search (BM25 + semantic + reranking)"""
```

**HybridSearcher (RRF Fusion)**
```python
class HybridSearcher:
    def search(self, query: str, strategy: str, top_k: int,
               alpha: float) -> Dict[str, Any]:
        """Combine semantic and keyword search with RRF"""
```

**CrossEncoderReranker (Relevance Reranking)**
```python
class CrossEncoderReranker:
    def rerank(self, query: str, results: List[Dict], 
               top_k: int) -> Dict[str, Any]:
        """Rerank results using cross-encoder scores"""
```

**AnswerGenerator (RAG Generation)**
```python
class AnswerGenerator:
    def generate_answer(self, query: str, 
                        retrieved_results: List[Dict]) -> Dict[str, Any]:
        """Generate natural language answer with citations"""
```

---

## Docker Architecture

### Multi-Container Setup

```
┌─────────────────────────────────────────────────┐
│ Docker Compose Network (automatic)              │
│                                                 │
│  ┌─────────────┐         ┌──────────────────┐ │
│  │   ollama    │◄────────│  rag-pipeline    │ │
│  │             │         │                  │ │
│  │ - Llama 3.2 │ API     │ - RAG code       │ │
│  │ - Port 11434│ calls   │ - CLI            │ │
│  │ - Healthchk │         │ - Queries Ollama │ │
│  └─────────────┘         └──────────────────┘ │
│        │                         │             │
│        ▼                         ▼             │
│  [ollama-models]       [Host directories]     │
│   Docker volume         - data/processed      │
│   ~2 GB persisted       - data/vector_store   │
│                         - logs/               │
└─────────────────────────────────────────────────┘
```

### Service Definitions

**Ollama Service**
- **Image:** `ollama/ollama:0.32.0` (pinned in docker-compose.yml)
- **Purpose:** Run local LLM
- **Volume:** `ollama-models` (persists ~2 GB model)
- **Healthcheck:** `ollama list || exit 1` every 10s
- **Network:** Internal only (no port exposure)

**RAG Pipeline Service**
- **Build:** From local `Dockerfile`
- **Depends on:** `ollama` (waits for healthy)
- **Connects to:** `http://ollama:11434`
- **Volumes:**
  - `./data/processed` → Processed documents
  - `./data/state` → State tracking
  - `./data/vector_store` → ChromaDB
  - `./logs` → Application logs
  - `huggingface-cache` → Model cache (embedder + reranker, mounted at HF_HOME)

### Service Communication

```
RAG Pipeline Container
    ↓ HTTP
http://ollama:11434/api/generate
    ↓
Ollama Container
    ↓ Docker network DNS
Service name "ollama" resolves to container IP
```

Docker Compose automatically creates an internal network where services reference each other by name.

### Volume Management

**Named Volumes (persistent in Docker):**
1. `huggingface-cache` (~170 MB) - Embedding + reranker models (mounted at HF_HOME)
2. `ollama-models` (~2 GB) - LLM models

**Bind Mounts (on host filesystem):**
1. `./data/processed` - JSON documents (~4 MB)
2. `./data/state` - State files (~12 KB)
3. `./data/vector_store` - ChromaDB (~86 MB)
4. `./logs` - Application logs

**Why this design:**
- Models cached in volumes (survive `docker compose down`)
- Data on host (easy to inspect, backup, version control)
- Logs on host (easy to tail, analyze)

---

## Storage Architecture

### File System Layout

```
rag_pipeline/
├── data/
│   ├── corpus/                    # [Tracked in Git]
│   │   └── scikit-learn-1.7.2-docs/
│   │       └── 420 HTML files
│   │
│   ├── processed/                 # [Generated, gitignored]
│   │   ├── api/
│   │   │   └── 251 JSON files (~2.5 MB)
│   │   ├── guide/
│   │   │   └── 46 JSON files (~0.5 MB)
│   │   ├── example/
│   │   │   └── 82 JSON files (~0.8 MB)
│   │   └── other/
│   │       └── 37 JSON files (~0.2 MB)
│   │
│   ├── vector_store/             # [Generated, gitignored]
│   │   ├── chroma.sqlite3        # SQLite database
│   │   ├── <uuid-fixed>/         # Fixed collection
│   │   ├── <uuid-semantic>/      # Semantic collection
│   │   └── <uuid-hierarchical>/  # Hierarchical collection
│   │   Total: ~86 MB
│   │
│   ├── state/                    # [Generated, gitignored]
│   │   ├── preprocessing_state.json
│   │   └── indexing_state.json
│   │   Total: ~12 KB
│   │
│   └── evaluation/
│       ├── test_set.json         # [Tracked in Git]
│       └── results/              # [Generated, gitignored]
│           └── evaluation_*.json
│
└── logs/                          # [Generated, gitignored]
    ├── preprocessing.log
    ├── indexing.log
    ├── query.log
    └── evaluation.log
```

### Storage Requirements

| Component | Size | Location | Tracked in Git? |
|-----------|------|----------|-----------------|
| Source corpus | ~15 MB | `data/corpus/` | Yes |
| Processed JSON | ~4 MB | `data/processed/` | No (generated) |
| Vector store | ~86 MB | `data/vector_store/` | No (generated) |
| State files | ~12 KB | `data/state/` | No (runtime) |
| Test set | ~12 KB | `data/evaluation/test_set.json` | Yes |
| Logs | Variable | `logs/` | No (runtime) |
| **Total (tracked)** | **~15 MB** | | |
| **Total (generated)** | **~90 MB** | | |

### Database Schema

**ChromaDB Collections:**

Each collection stores:
- **Documents:** Chunk content (text)
- **Embeddings:** 384-dimensional vectors
- **Metadata:** `doc_id`, `chunk_index`, `doc_type`, `strategy`, etc.
- **IDs:** Unique chunk identifiers

Collections are independent - allows querying single or multiple strategies.

---

## Performance Characteristics

### End-to-End Timing

| Operation | First Run | Cached Run | Notes |
|-----------|-----------|------------|-------|
| **Setup (`./setup.sh`)** | **5-10 min** | **n/a** | One-time |
| - Build images | 2-3 min | ~30s | Docker layer caching |
| - Download LLM | 2-5 min | instant | Cached in volume |
| - Preprocess | 30-40s | <1s | State cached |
| - Index | 90-100s | ~5s | State cached |
| **Query (retrieval)** | ~5-6s | ~1.2s | First query loads model |
| **Query (+ generation)** | ~10-15s | ~3-4s | LLM + retrieval |
| **Evaluation (full)** | ~45 min | ~45 min | No caching (by design) |

### Component Performance (CPU, Intel i7/similar)

**Preprocessing:**
- 420 HTML files → JSON
- Time: ~35 seconds
- Throughput: ~12 files/second

**Chunking:**
- 420 docs → ~4,000 chunks
- Time: ~5-10 seconds total
- Breakdown: Fixed (3s), Semantic (2s), Hierarchical (4s)

**Embedding:**
- ~4,000 chunks → 384-dim vectors
- Time: ~70-80 seconds (CPU)
- Throughput: ~50 chunks/second
- GPU would be 5-10x faster

**Retrieval:**
- Query → Top-20 results
- Time: ~1.2 seconds
- Breakdown: Embed query (0.04s), Search (1.15s), Format (0.01s)

**Generation:**
- Retrieved chunks → Natural answer
- Time: ~2-3 seconds (3B model, CPU)
- Breakdown: Prompt (0.01s), LLM (2.5s)

### Scalability Projections

**Current scale** (420 docs, ~4,000 chunks):
- Query latency: ~1.2s
- Memory: ~500 MB

**10x scale** (~4,200 docs, ~40,000 chunks):
- Query latency: ~2-3s (ChromaDB efficient with HNSW)
- Memory: ~1 GB

**Optimization opportunities:**
- GPU acceleration (5-10x faster embeddings)
- ANN indices (faster similarity search)
- Result caching (instant for repeated queries)
- Batch query processing

---

## State Management

### State Files

**preprocessing_state.json:**
```json
{
  "pruning_completed": true,
  "pruning_timestamp": "2025-10-19T20:28:15.123456",
  "processing_completed": true,
  "processing_timestamp": "2025-10-19T20:28:30.789012",
  "files_processed": 420,
  "files_by_type": {
    "api": 251,
    "guide": 46,
    "example": 82,
    "other": 37
  }
}
```

**indexing_state.json:**
```json
{
  "fixed": {
    "indexed": true,
    "chunk_count": 1595,
    "doc_count": 420,
    "timestamp": "2025-10-19T20:28:45.123456"
  },
  "semantic": { ... },
  "hierarchical": { ... }
}
```

### State Validation

**Preprocessing validation:**
1. Check if state file exists
2. If exists, verify `processing_completed: true`
3. Validate files exist in `data/processed/`
4. Verify file count matches state

**Indexing validation:**
1. Check if state file exists for strategy
2. Verify `indexed: true`
3. Check ChromaDB collection exists
4. Verify chunk count matches (exact, not approximate)

**Benefits:**
- Saves 35s on subsequent preprocessing runs
- Saves 90s on subsequent indexing runs
- Prevents accidental re-processing
- `--force` flag overrides when needed

---

## Configuration Architecture

### Centralized YAML Configuration

**Location:** `config/config.yaml`

**Design pattern:**
```python
# Access via dot notation
config.get('paths.corpus_root')
config.get('chunking.fixed.chunk_size', default=512)

# Path resolution (relative to project root)
path = config.get_path('paths.vector_store_dir', create=True)
```

**Key sections:**
- `paths.*` - File system paths
- `preprocessing.*` - HTML parsing settings
- `chunking.*` - Strategy configurations
- `embeddings.*` - Model selection
- `retrieval.*` - Search parameters
- `generation.*` - LLM settings
- `evaluation.*` - Metrics configuration

**Benefits:**
- No hardcoded values in code
- Easy experimentation (just edit YAML)
- Self-documenting (comments in config)
- Environment-agnostic (paths resolve automatically)

---

## Summary

This architecture demonstrates:
- Modularity: 27 components with clear interfaces
- Scalability: ChromaDB + state management handle growth
- Reproducibility: Docker ensures consistent environment
- Flexibility: Configuration-driven, easy to swap components
- Practical engineering: Caching, logging, error handling, validation

**For detailed implementation:** See [CHANGELOG.md](CHANGELOG.md) for complete development history (Iterations 1-5)

**For design rationale:** See [DESIGN.md](DESIGN.md)
