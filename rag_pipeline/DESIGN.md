# Design Principles & Trade-offs

**Version:** 1.3  
**Last Updated:** July 2026

This document explains the design philosophy behind the RAG Pipeline, key architectural decisions, and trade-offs made throughout development.

---

## Table of Contents

1. [Core Design Principles](#core-design-principles)
2. [Architectural Decisions](#architectural-decisions)
3. [Trade-offs Analysis](#trade-offs-analysis)
4. [What Would Change for Production](#what-would-change-for-production)
5. [Learning Outcomes](#learning-outcomes)

---

## Core Design Principles

### 1. Modularity

**Principle:** Each component is independently testable and swappable.

**Implementation:**
- Abstract base classes (e.g., `BaseChunker`)
- Clear interfaces between modules
- Dependency injection (config, embedder, vector store passed as arguments)

**Example:**
```python
class BaseChunker(ABC):
    @abstractmethod
    def chunk_document(self, document: Dict) -> List[Chunk]:
        pass
```

Any new chunking strategy just inherits and implements this interface.

**Benefits:**
- Easy to add new chunking strategies
- Can swap embedding models via config
- Each module can be tested independently
- Reduces coupling between components

---

### 2. Configuration-Driven

**Principle:** No hardcoded values; everything configurable via YAML.

**Implementation:**
- Single `config/config.yaml` file
- Dot-notation access: `config.get('chunking.fixed.chunk_size')`
- Path resolution relative to project root
- Defaults provided in code

**Example:**
```yaml
embeddings:
  model: "all-MiniLM-L6-v2"  # Easy to change!
  batch_size: 32
  device: "cpu"
```

To use a different model:
```yaml
model: "all-mpnet-base-v2"  # Just edit this line
```

**Benefits:**
- Experimentation without code changes
- Self-documenting configuration
- Environment-agnostic (dev vs prod configs)
- Easy for reviewers to understand settings

---

### 3. State Management

**Principle:** Intelligent caching prevents redundant operations.

**Implementation:**
- JSON state files track completion
- Validation before skipping (exact count checks)
- `--force` flag to override when needed
- State includes timestamps and statistics

**Example:**
```python
if preprocessing_already_done() and not force_reprocess:
    logger.info("Preprocessing already complete (use --force to rerun)")
    return
```

**Benefits:**
- Saves 35s on preprocessing runs
- Saves 90s on indexing runs
- Better developer experience
- Prevents accidental data loss

---

### 4. Proper Logging

**Principle:** Structured logging for debugging and monitoring.

**Implementation:**
- Python's built-in `logging` module
- Per-module loggers
- Both console and file output
- Configurable log levels
- Timestamps and context

**Example:**
```python
logger = get_logger("query_processor")
logger.info(f"Processing query: '{query_text}'")
logger.debug(f"Query embedding shape: {embedding.shape}")
logger.error(f"Query failed: {error}", exc_info=True)
```

**Benefits:**
- Essential for debugging
- Production-ready monitoring
- Easy to trace issues
- No print() statements cluttering code

---

### 5. Type Hints

**Principle:** Full type annotations for better code quality.

**Implementation:**
- Type hints on all function signatures
- Return type annotations
- Complex types (Union, Optional, Dict, List)
- Enables IDE autocomplete and type checking

**Example:**
```python
def process_query(
    self,
    query_text: str,
    strategy: Optional[str] = None,
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
```

**Benefits:**
- Catches bugs before runtime
- Better IDE support
- Self-documenting code
- Easier for others to understand

---

### 6. Documentation

**Principle:** Clear docstrings and comprehensive documentation.

**Implementation:**
- Google-style docstrings for all classes/functions
- README for quick start
- Iteration summaries for detailed implementation
- Architecture and design docs

**Example:**
```python
def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
    """
    Chunk a single document using fixed-size strategy.
    
    Args:
        document: Document dictionary with 'content' and metadata
        
    Returns:
        List of Chunk objects with uniform size
        
    Raises:
        ValueError: If document is missing required fields
    """
```

**Benefits:**
- Easy onboarding for new developers
- Future maintainability
- Reviewers can understand quickly

---

### 7. Reproducibility

**Principle:** Docker ensures consistent environment across machines.

**Implementation:**
- Multi-container setup (Ollama + RAG pipeline)
- One-command setup script
- Models cached in Docker volumes
- No host dependencies except Docker

**Benefits:**
- "Works on my machine" → "Works everywhere"
- No manual Ollama installation
- No Python environment issues
- Reviewers can test immediately

---

### 8. Error Handling and Validation

**Principle:** Graceful failures with informative error messages.

**Implementation:**
- Try-catch blocks with informative errors
- Input validation before processing
- Idempotent operations (can run multiple times safely)
- Graceful degradation (continue on non-fatal errors)

**Example:**
```python
try:
    processor.run(force_reprocess=force_reprocess)
except FileNotFoundError as e:
    logger.error(f"Corpus not found: {e}")
    print("Error: Please ensure corpus is in data/corpus/")
    sys.exit(1)
except Exception as e:
    logger.error(f"Preprocessing failed: {e}", exc_info=True)
    sys.exit(1)
```

**Benefits:**
- System doesn't crash on edge cases
- Clear error messages guide users
- Safe to re-run commands
- More reliable operation

---

## Architectural Decisions

### Major Technology Choices

| Decision | Alternatives Considered | Chosen Approach | Rationale |
|----------|------------------------|-----------------|-----------|
| **Vector DB** | Pinecone, Weaviate, Qdrant, FAISS | ChromaDB | Local, persistent, good for MVP. No API keys needed. |
| **Embeddings** | OpenAI, Cohere, custom models | sentence-transformers | Free, local, industry standard. Easy model swapping. |
| **LLM** | OpenAI, Anthropic, Cohere | Ollama + Llama 3.2 | Free, local, no API keys. Reproducible for reviewers. |
| **Chunking** | Just one strategy | Three strategies | Enables empirical comparison. Shows understanding of trade-offs. |
| **Hybrid Search** | Semantic only | BM25 + semantic with RRF | Combines precision of keywords with semantic understanding. |
| **Query Enhancement** | None | LLM-based rewriting with caching | Improves retrieval quality; caching avoids repeated LLM calls. |
| **Reranking** | None | Cross-encoder (ms-marco-MiniLM) | Higher precision; lazy-loaded to minimize startup overhead. |
| **Config** | JSON, TOML, environment variables | YAML | Human-readable, supports comments, widely used. |
| **State** | Database, Redis, nothing | JSON files | Simple, transparent, easy to inspect and debug. |
| **Deployment** | Kubernetes, serverless, manual | Docker Compose | Appropriate for portfolio. Multi-container support. |
| **Testing** | Comprehensive unit tests | End-to-end validation + evaluation metrics | Pragmatic for MVP. Evaluation provides systematic validation. |

---

## Trade-offs Analysis

### 1. ChromaDB vs Production Vector Databases

**Chosen: ChromaDB**

**Pros:**
- Simple setup (no external service)
- Persistent (survives restarts)
- Good enough for 4000 vectors
- Local (no network latency)

**Cons:**
- Not as scalable as Pinecone/Weaviate
- No distributed deployment
- Limited query optimization
- Not managed (no monitoring dashboard)

**Alternative: Pinecone**

**Pros:**
- Highly scalable (billions of vectors)
- Managed service (no ops burden)
- Advanced features (namespaces, metadata filtering)
- Built-in monitoring

**Cons:**
- Requires account and API key
- Costs money after free tier
- Not local (network dependency)
- Less reproducible for portfolio reviewers

**Justification:** For a portfolio project demonstrating RAG concepts, simplicity and reproducibility trump scale. A reviewer shouldn't need to create accounts or configure cloud services to test the project.

**When to switch:** If scaling beyond ~100K documents or deploying to production.

---

### 2. Local LLM (Ollama) vs Cloud APIs

**Chosen: Ollama + Llama 3.2**

**Pros:**
- Completely free (no API costs)
- Private (data stays local)
- Offline-capable
- Reproducible (no API key management)
- Docker-deployable

**Cons:**
- Slower than cloud APIs (~2-3s vs ~0.5s)
- Lower quality than GPT-4/Claude
- Requires hardware (4GB RAM for 3B model)
- First-time model download (~2GB)

**Alternative: OpenAI GPT-4**

**Pros:**
- Much higher quality answers
- Faster response times
- Supports longer contexts
- Better instruction following

**Cons:**
- Costs money ($0.01-0.03 per query)
- Requires API key (barrier for reviewers)
- Network dependency
- Privacy concerns (data sent to OpenAI)

**Justification:** For a portfolio project, the ability for anyone to clone and run without API keys is more valuable than answer quality. The system architecture is the same whether using Ollama or OpenAI - just swap the LLM client.

**When to switch:** Production deployment where answer quality is critical.

---

### 3. Three Chunking Strategies vs One

**Chosen: Three strategies (fixed, semantic, hierarchical)**

**Pros:**
- Enables empirical comparison
- Shows understanding of trade-offs
- Demonstrates systematic evaluation
- More interesting for portfolio

**Cons:**
- More implementation complexity
- 3x storage space
- Longer indexing time
- More code to maintain

**Alternative: Just fixed chunking**

**Pros:**
- Simpler implementation
- Faster indexing
- Less storage
- Easier to understand

**Cons:**
- Misses learning opportunity
- No comparison to show best approach
- Less impressive for portfolio
- Can't demonstrate evaluation methodology

**Justification:** The whole point of this project is to go beyond simple demos. Implementing multiple strategies and systematically comparing them demonstrates critical thinking about design choices, ability to evaluate approaches empirically, and understanding that the best choice depends on context.

This is what separates a basic portfolio project from a more thorough one.

**When to reconsider:** If only one strategy is needed for a specific production use case (but keep evaluation framework to validate the choice).

---

### 4. Manual Testing vs Unit Tests

**Chosen: Manual end-to-end testing + systematic evaluation metrics**

**Pros (current approach):**
- Faster to implement for MVP
- Real-world validation
- Evaluation framework provides comprehensive validation
- End-to-end confidence

**Cons (current approach):**
- No automated regression testing
- Harder to catch edge cases
- No CI/CD integration
- Less professional appearance

**Alternative: Comprehensive pytest coverage**

**Pros:**
- Automated regression detection
- Catch edge cases early
- CI/CD integration
- More professional

**Cons:**
- Time-consuming for MVP
- Still need end-to-end testing
- Mocking LLMs is tricky
- May not find integration issues

**Justification:** For a portfolio project with limited time, demonstrating the system works end-to-end is more important than having 80% test coverage. The evaluation framework with 35 test questions provides systematic validation that's more relevant for RAG systems.

**Compromise:** Add 5-10 representative unit tests to show understanding of testing patterns:
- Config loading and path resolution
- Chunking strategies with known inputs
- Metric calculations (Recall@k, MRR with known data)
- Prompt building

**Outcome (update):** the compromise outgrew the original decision — the project
now has a 101-test unit suite (LLM calls mocked) that runs in a path-filtered
GitHub Actions workflow on every push, installing from the pinned
`requirements.lock`. The "manual testing only" choice above is superseded; the
evaluation framework remains the primary end-to-end validation.

This demonstrates testing ability without being exhaustive.

---

### 5. State Tracking vs Always Reprocess

**Chosen: State tracking with JSON files**

**Pros:**
- Saves 35s on preprocessing
- Saves 90s on indexing
- Better developer experience
- Prevents accidental data loss

**Cons:**
- Additional complexity
- State files can become stale
- Need validation logic
- Debugging state issues

**Alternative: Always reprocess**

**Pros:**
- Simpler code (no state management)
- Always fresh data
- No state synchronization issues

**Cons:**
- Wastes 2 minutes on every run
- Frustrating during development
- Unnecessary computation
- Higher Docker resource usage

**Justification:** Developer experience matters, even in a portfolio project. Waiting 2 minutes every time you test a small change is frustrating. The `--force` flag provides an escape hatch when needed.

**Key insight:** This demonstrates understanding that systems need to be efficient, not just correct.

---

### 6. Hybrid Search vs Pure Semantic Search

**Chosen: Hybrid search (BM25 + semantic with RRF)**

**Pros:**
- Better handling of exact technical terms (e.g., "fit_transform", "GridSearchCV")
- Combines precision of keyword matching with semantic understanding
- Configurable alpha parameter for tuning
- Industry-standard approach for technical documentation

**Cons:**
- More complex implementation
- Requires maintaining BM25 index alongside vector store
- Slightly higher latency (~20ms overhead)
- More configuration options to tune

**Alternative: Pure semantic search**

**Pros:**
- Simpler implementation
- No additional index to maintain
- Works well for natural language queries

**Cons:**
- Struggles with exact technical terms
- May miss documents with exact keyword matches
- Less precise for code-related queries

**Justification:** For technical documentation Q&A (especially sklearn APIs), users often search for exact function names. Pure semantic search can miss these because the embedding model focuses on semantic meaning rather than exact tokens. Hybrid search combines the best of both approaches.

---

### 7. Cross-Encoder Reranking vs No Reranking

**Chosen: Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)**

**Pros:**
- Higher precision in top results
- Jointly scores query-document pairs (more accurate than bi-encoder)
- Lazy-loaded to avoid startup overhead
- Graceful fallback if reranking fails

**Cons:**
- Added latency (~100-200ms)
- Additional model to load/maintain
- Increased memory usage
- More complex pipeline

**Alternative: No reranking**

**Pros:**
- Lower latency
- Simpler implementation
- Less memory usage

**Cons:**
- Lower precision in top results
- Bi-encoder similarity may not reflect true relevance
- Misses opportunity for quality improvement

**Justification:** For RAG systems, the quality of top retrieved documents directly impacts answer quality. The cross-encoder's ~100ms latency is acceptable for the precision improvement, especially since the LLM generation step takes 2-3 seconds anyway.

---

### 8. LLM Query Rewriting vs No Rewriting

**Chosen: LLM-based query rewriting with caching**

**Pros:**
- Expands abbreviations (PCA → Principal Component Analysis)
- Adds relevant synonyms for better recall
- Removes conversational filler
- Caching avoids repeated LLM calls

**Cons:**
- Added latency for cache misses (~500ms)
- Depends on LLM availability
- May occasionally produce poor rewrites

**Mitigations:**
- LRU cache (128 entries) for repeated queries
- Graceful fallback to original query on failure
- Low temperature (0.3) for deterministic rewrites

**Justification:** Query rewriting improves retrieval quality by bridging the vocabulary gap between user queries and document content. The caching strategy minimizes the latency impact for repeated or similar queries.

---

## What Would Change for Production

This section acknowledges that portfolio projects and production systems have different requirements.

### Performance Considerations

**Embedding Model Loading:**
The current CLI implementation loads the embedding model fresh for each query command. 
This adds ~1-2 seconds of startup overhead per query. For production deployments:
- Implement a server mode that keeps the model in memory
- Use a model server (e.g., Triton, TensorFlow Serving)
- Or pre-warm the model in a long-running process

**Current behavior:** Acceptable for CLI/batch usage, not suitable for low-latency API serving.

### Infrastructure Changes

| Component | Portfolio | Production | Reason |
|-----------|-----------|-----------|--------|
| **Vector DB** | ChromaDB | Pinecone/Weaviate | Scale to millions of documents |
| **LLM** | Ollama (local) | OpenAI/Claude API | Higher quality answers |
| **Hosting** | Docker Compose | Kubernetes | Auto-scaling, high availability |
| **Config** | YAML file | Environment vars + Secrets Manager | Security, multi-environment |
| **Logging** | File + console | Structured logs → ELK/Datadog | Centralized monitoring |
| **State** | JSON files | Database (PostgreSQL) | Concurrent access, ACID properties |

### Additional Features for Production

1. **Monitoring & Alerting**
   - Prometheus metrics (query latency, error rates)
   - Grafana dashboards
   - PagerDuty alerts for failures

2. **Authentication & Authorization**
   - API keys for access control
   - Rate limiting per user
   - Usage tracking and billing

3. **Caching Layer**
   - Redis for query results
   - Cache hit rate monitoring
   - TTL-based invalidation

4. **Testing & CI/CD**
   - Integration tests
   - Performance/load tests
   - Deployment pipelines
   - (Unit tests and GitHub Actions CI are already in place — 111 tests run on
     every push from the pinned lock file)

5. **Error Recovery**
   - Retry logic with exponential backoff
   - Circuit breakers for external services
   - Dead letter queues for failed operations

6. **Performance Optimization**
   - Async query processing
   - Batch embedding generation
   - Query result pre-computation for common queries
   - CDN for static assets

7. **Security Hardening**
   - Input sanitization
   - SQL injection prevention
   - Rate limiting
   - DDoS protection

### Cost Considerations

| Component | Portfolio Cost | Production Cost (Monthly) |
|-----------|----------------|--------------------------|
| Compute | $0 (local) | $200-500 (Kubernetes) |
| Vector DB | $0 (ChromaDB) | $100-300 (Pinecone) |
| LLM API | $0 (Ollama) | $500-2000 (OpenAI, varies with usage) |
| Monitoring | $0 | $50-200 (Datadog) |
| **Total** | **$0** | **$850-3000+** |

**Justification for portfolio approach:** Zero cost means anyone can test it. The architecture is the same - just swap components.

---

## Learning Outcomes

### What This Project Demonstrates

**1. RAG System Understanding**
- Chunking strategies and their trade-offs
- Embedding generation and vector similarity
- Retrieval strategies (dense, multi-collection)
- Prompt engineering for RAG
- Citation extraction and source tracking

**2. ML System Engineering**
- State management and caching
- Configuration-driven architecture
- Evaluation methodology (not just implementation)
- Performance optimization considerations
- Trade-offs between approaches

**3. Software Engineering**
- Clean code architecture
- Modularity and abstractions
- Type safety with Python type hints
- Proper error handling
- Professional documentation

**4. DevOps & Deployment**
- Docker containerization
- Multi-service orchestration (Docker Compose)
- Volume management for persistence
- Healthchecks and service dependencies
- Automated setup scripts

**5. Critical Thinking**
- Recognizing when LLM judging wasn't working
- Pivoting to objective retrieval metrics
- Documenting trade-offs honestly
- Understanding portfolio vs production differences

### The Key Differentiator: Evaluation (Iteration 5)

**Why evaluation matters for this project:**

Most portfolio RAG projects show:
- Implementation of retrieval
- Integration with LLM
- But no systematic evaluation
- No proof of what works better

This project shows:
- Implementation (iterations 1-4)
- Quantitative evaluation (iteration 5)
- Evidence-based conclusion: Fixed chunking wins
- Quantitative comparison across 35 test questions — isolating chunking under
  semantic-only retrieval (no hybrid/rewriting/reranking), with small
  between-strategy gaps read as directional rather than significance-tested

**This demonstrates:**
- Scientific thinking
- Data-driven decision making
- Ability to measure and compare approaches
- Understanding that implementation is just the first step

**Example of pragmatic decision-making:**
> "I initially implemented LLM-as-judge for answer quality, but Llama 3.2 3B lacked discriminative capability. Rather than report misleading metrics, I pivoted to focus on retrieval quality (Recall@k, MRR, NDCG) which are objective and reproducible."

This honesty and pragmatism demonstrates mature engineering judgment.

---

## Summary

### Design Philosophy

This project follows the principle: "Sound engineering practices within portfolio constraints"

- Sound engineering: State management, logging, error handling, evaluation
- Portfolio constraints: Local deployment, free tools, manageable scope

### Key Insights

1. Systematic evaluation separates simple demos from more complete systems
2. Reproducibility (Docker, local LLM) is critical for portfolio projects
3. Trade-offs should be documented, not hidden
4. Code quality matters as much as ML knowledge
5. Honesty about limitations builds credibility

---

**For implementation details:** See [ARCHITECTURE.md](ARCHITECTURE.md)  
**For quick start:** See [README.md](README.md)  
**For iteration details:** See [CHANGELOG.md](CHANGELOG.md)
