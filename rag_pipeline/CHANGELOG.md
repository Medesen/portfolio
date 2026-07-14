# Changelog

All notable changes to the RAG Pipeline project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.0] - 2025-12-05 (Advanced Retrieval Features)

### Added

**Hybrid Search with RRF:**
- Implemented Reciprocal Rank Fusion (RRF) combining BM25 keyword search with semantic embeddings
- BM25 index using rank-bm25 with Porter stemming and stopword removal
- Configurable alpha parameter for semantic/keyword balance (default: 0.7 = 70% semantic)
- Three search modes: `hybrid`, `semantic`, `keyword`
- New module: `src/retrieval/bm25_index.py` - BM25 inverted index implementation
- New module: `src/retrieval/hybrid_searcher.py` - RRF fusion logic

**Query Rewriting:**
- LLM-based query rewriting for improved retrieval quality
- Expands abbreviations (e.g., "PCA" → "Principal Component Analysis PCA")
- Adds scikit-learn synonyms and technical terms
- Removes conversational filler words
- LRU caching to avoid redundant LLM calls (configurable cache size: 128)
- Graceful fallback to original query on LLM failure
- New module: `src/retrieval/query_rewriter.py`

**Cross-Encoder Reranking:**
- Cross-encoder reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Over-fetches 50 candidates, reranks, returns top 10
- Lazy model loading to avoid startup overhead
- Graceful fallback if reranking fails
- New module: `src/retrieval/reranker.py`

**Benchmark Script:**
- Script to benchmark different overfetch_k values for reranking
- Measures latency vs quality trade-offs
- Usage: `make benchmark ARGS="--overfetch-values 30 50 60"`
- New script: `scripts/benchmark_overfetch.py`

**Unit Tests:**
- Added 75 new tests (93 total, up from 18)
  - `tests/test_hybrid_search.py` - BM25, RRF fusion, alpha weighting (24 tests)
  - `tests/test_query_rewriter.py` - LLM integration, caching, fallback (23 tests)
  - `tests/test_reranker.py` - Score reordering, fallback behavior (18 tests)
  - `tests/test_bm25_strategy_switching.py` - Strategy loading (10 tests)

### Changed
- `QueryProcessor` now integrates hybrid search, query rewriting, and reranking
- Makefile: `make query` now includes LLM generation by default
- Makefile: Added `make query-retrieve` for retrieval-only (no LLM)
- Makefile: Added `make benchmark` for reranking benchmarks
- Configuration: Added `query_rewriting` and `reranking` sections to config.yaml

### Configuration Options
```yaml
# Hybrid search
retrieval:
  search_mode: "hybrid"  # "semantic", "keyword", or "hybrid"
  hybrid_alpha: 0.7      # 1.0 = pure semantic, 0.0 = pure keyword
  rrf_k: 60              # RRF dampening factor

# Query rewriting
query_rewriting:
  enabled: true
  temperature: 0.3
  max_tokens: 100
  cache_size: 128

# Reranking
reranking:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  overfetch_k: 50
  final_top_k: 10
```

---

## [1.0.0] - 2025-10-21

### ✅ Project Complete - All 5 Iterations Finished

**Major milestone:** Production-ready RAG system with systematic evaluation proving fixed chunking wins for technical documentation.

---

## [1.1.0] - 2025-10-21 (Portfolio Preparation)

### Added

**Unit Testing Infrastructure:**
- Created comprehensive test suite with 18 unit tests demonstrating testing patterns
  - `tests/test_config.py` - Configuration loading and path resolution (4 tests)
  - `tests/test_chunking.py` - Fixed and semantic chunking strategies (4 tests)  
  - `tests/test_metrics.py` - Recall@k, MRR, NDCG calculations (6 tests)
  - `tests/test_embedder.py` - Embedding generation with mocking (4 tests)
  - `tests/conftest.py` - Pytest fixtures for reusable test data
- Added `pytest>=7.4.0` and `pytest-cov>=4.1.0` to requirements.txt
- All tests run via Docker: `make test`, `make test-cov`, `make test-file F=<file>`
- Test coverage: 21% (focused on core components)

**Makefile Build System:**
- Added comprehensive `Makefile` with 30+ command shortcuts
  - Setup: `make setup`, `make build`, `make up`, `make down`
  - Testing: `make test`, `make test-cov`, `make test-file`
  - Operations: `make preprocess`, `make index`, `make query`, `make eval`
  - Utilities: `make status`, `make logs`, `make shell`
  - Cleanup: `make clean`, `make clean-data`, `make clean-all`
- Built-in help system: `make help` shows all commands with examples
- Cross-platform compatible (Linux, macOS, Windows with make installed)

**Windows Support:**
- Added `setup.ps1` PowerShell script for native Windows setup
- No Git Bash or WSL2 required (though still supported as alternative)
- Updated README with Windows-specific installation instructions
- All Docker commands work identically across platforms

**Documentation:**
- Consolidated to 4 core markdown files for portfolio clarity:
  - `README.md` - Quick start and user guide
  - `ARCHITECTURE.md` - System design and technical details
  - `DESIGN.md` - Design principles and trade-offs
  - `CHANGELOG.md` - Complete version history (this file)
- Removed 20+ redundant development status and iteration documents
- Cleaner repository structure for hiring manager review

### Changed
- **README.md:** Now uses `make` commands throughout (simpler than raw Docker commands)
- **README.md:** Enhanced FAQ with Windows make installation instructions
- **README.md:** Updated cross-platform compatibility section
- **README.md:** Added dedicated Testing section with Docker-only approach explanation
- **ARCHITECTURE.md:** References consolidated documentation structure
- **Test commands:** Must use `--entrypoint pytest` when using raw Docker (not needed with Makefile)

### Fixed
- Test imports corrected to use `RetrievalMetrics` class (not standalone functions)
- Test config paths updated to match actual YAML structure (`chunking.strategies.fixed.*`)
- All 18 unit tests now pass ✅

### Notes
- This version focuses on portfolio presentation quality
- Systematic evaluation framework (35 questions) remains primary validation mechanism
- Unit tests demonstrate patterns without being exhaustive (intentional for portfolio scope)

---

## Iteration 5: Evaluation Framework - 2025-10-20

### Added
- **35-question curated test set** (`data/evaluation/test_set.json`)
  - 10 factual questions (e.g., "What does StandardScaler do?")
  - 10 how-to questions (e.g., "How do I use GridSearchCV?")
  - 8 comparison questions (e.g., "StandardScaler vs RobustScaler?")
  - 7 troubleshooting questions (e.g., "Why am I getting NaN values?")
- **Comprehensive retrieval metrics:**
  - Recall@k (k=5, 10, 20) - Coverage of relevant documents
  - MRR (Mean Reciprocal Rank) - Ranking quality
  - NDCG@k - Position-weighted relevance
  - Topic Coverage - Semantic keyword matching
- **Evaluation modules:**
  - `src/evaluation/evaluator.py` - Main orchestrator (511 lines)
  - `src/evaluation/metrics.py` - Metric implementations (272 lines)
  - `src/evaluation/llm_judge.py` - LLM-based judging (optional) (518 lines)
  - `src/evaluation/test_loader.py` - Test set loading (105 lines)
  - `src/evaluation/results_analyzer.py` - Statistical analysis (181 lines)
- **CLI command:** `evaluate --strategy STRAT --max-questions N --no-judge --report`
- **Automation script:** `scripts/run_evaluation.py`
- **Analysis notebook:** `notebooks/evaluation_analysis.ipynb`

### Results
- **Winner: Fixed chunking**
  - Recall@10: **0.51** (best)
  - MRR: **0.51** (best)
  - NDCG@10: **0.40** (best)
- Semantic: Recall@10: 0.46, MRR: 0.48, NDCG@10: 0.36
- Hierarchical: Recall@10: 0.50, MRR: 0.47, NDCG@10: 0.37
- All strategies achieve 90%+ topic coverage
- Scope: strategies compared under plain semantic retrieval (no hybrid
  BM25, query rewriting, or reranking) to isolate the chunking variable

### Changed
- **Critical design decision:** Pivoted from LLM-based answer judging to retrieval-focused evaluation
  - Reason: Llama 3.2 3B lacked discriminative capability (scores 3.5-4.0 for all answers)
  - Solution: Focus on objective retrieval metrics
  - Impact: More reliable and reproducible evaluation
- Set `evaluation.judge_answers: false` in config (by design)

### Documentation
- Added ITERATION_5_SUMMARY.md (309 lines)
- Updated README.md with evaluation results
- Updated PROJECT_DESCRIPTION.md (Iteration 5 section complete)

---

## Iteration 4: Answer Generation & Docker - 2025-10-19

### Added
- **LLM integration** via Ollama
  - `src/generation/llm_client.py` - Ollama API client (210 lines)
  - `src/generation/prompt_builder.py` - RAG prompt templates (180 lines)
  - `src/generation/answer_generator.py` - Generation pipeline (200 lines)
- **Multi-container Docker setup:**
  - Ollama service (separate container)
  - RAG pipeline service
  - Healthchecks for service dependencies
  - Named volumes for model persistence
- **Automated setup script:** `setup.sh` (75 lines)
  - One-command deployment: Build → Start Ollama → Download model → Preprocess → Index
- **CLI flag:** `--generate` for LLM-powered answers
- **Answer features:**
  - Natural language generation
  - Inline citations [1][2][3]
  - Source attribution
  - Metadata (timing, tokens, model info)

### Changed
- Ollama runs as Docker service (no manual installation)
- File permissions fixed (no root-owned files)
- Updated `config/config.yaml` with generation settings
- Modified `docker-compose.yml` for multi-service orchestration

### Documentation
- Added ITERATION_4_SUMMARY.md
- Added DOCKER_SETUP.md
- Added DOCKER_INTEGRATION_COMPLETE.md
- Added TESTING_ITERATION_4.md
- Updated README.md with Docker instructions

---

## Iteration 3: Query & Retrieval - 2025-10-18

### Added
- **Query processing infrastructure:**
  - `src/retrieval/query_processor.py` - Complete query pipeline (440 lines)
  - Natural language query input
  - Query embedding generation
  - Vector similarity search
  - Result merging across strategies
  - Deduplication by doc_id
- **CLI command:** `query "QUESTION" --strategy STRAT --top-k N --show-content --output FILE`
- **Result formatting:**
  - Rich console output with similarity scores
  - JSON export support
  - Configurable content display (excerpts vs full)
- **Multi-strategy querying:**
  - Query single strategy: `fixed`, `semantic`, `hierarchical`
  - Query all strategies: `all` (with merging)
  - Merge strategies: `interleave` or `top_scores`

### Changed
- Updated `config/config.yaml` with retrieval settings
- Enhanced `main.py` with query command

### Documentation
- Added ITERATION_3_SUMMARY.md
- Added TESTING_ITERATION_3.md
- Added IMPLEMENTATION_COMPLETE.md
- Updated README.md with query examples

---

## Iteration 2: Chunking & Indexing - 2025-10-17

### Added
- **Three chunking strategies:**
  - `src/chunking/fixed_chunker.py` - Fixed-size (512 tokens, 50 overlap)
  - `src/chunking/semantic_chunker.py` - Sentence/paragraph boundaries
  - `src/chunking/hierarchical_chunker.py` - Document structure-aware
  - `src/chunking/base_chunker.py` - Abstract base class
- **Embedding generation:**
  - `src/retrieval/embedder.py` - sentence-transformers wrapper (140 lines)
  - Model: `all-MiniLM-L6-v2` (384 dimensions)
  - Batch processing (32 chunks at a time)
- **Vector storage:**
  - `src/retrieval/vector_store.py` - ChromaDB wrapper (180 lines)
  - 3 separate collections (fixed, semantic, hierarchical)
  - Persistent storage in `data/vector_store/`
- **Indexing orchestration:**
  - `src/retrieval/indexer.py` - Build vector indices (250 lines)
  - State tracking for indexing (avoid re-indexing)
  - Validation with exact count checks
- **CLI command:** `index --strategy STRAT --force`

### Results
- Fixed: 1595 chunks, ~384 words each
- Semantic: 736 chunks, ~1000 words each
- Hierarchical: 1675 chunks, variable size
- Total: 4006 chunks across all strategies
- Vector store: ~86 MB

### Changed
- Updated `config/config.yaml` with chunking and embedding settings
- Enhanced `main.py` with index command

### Documentation
- Added ITERATION_2_SUMMARY.md
- Updated README.md with indexing instructions

---

## Iteration 1: Foundation & Preprocessing - 2025-10-16

### Added
- **Project structure:**
  - Created `src/` directory with module organization
  - Created `config/`, `data/`, `logs/`, `scripts/` directories
- **Configuration system:**
  - `src/utils/config.py` - YAML-based config loading (114 lines)
  - `config/config.yaml` - Centralized configuration
  - Dot-notation access: `config.get('paths.corpus_root')`
  - Automatic path resolution
- **Logging infrastructure:**
  - `src/utils/logger.py` - Structured logging (63 lines)
  - Console + file output
  - Per-module loggers
- **HTML preprocessing:**
  - `src/preprocessing/html_parser.py` - BeautifulSoup-based parsing (229 lines)
  - `src/preprocessing/corpus_processor.py` - Pipeline orchestration (240 lines)
  - Extracts clean content from HTML
  - Classifies document types (api, guide, example, other)
  - Preserves section structure
- **State management:**
  - JSON state files in `data/state/`
  - Idempotent operations (safe to re-run)
  - `--force` flag to override
- **CLI interface:**
  - `main.py` - argparse-based CLI (201 lines)
  - Command: `preprocess --force`
- **Corpus pruning:**
  - `scripts/prune_sklearn_corpus.py` - Reduce corpus to core topics (273 lines)

### Results
- 416 documents processed
  - API: 251 files (60.3%)
  - Guides: 46 files (11.1%)
  - Examples: 82 files (19.7%)
  - Other: 37 files (8.9%)
- Output: ~4 MB JSON files in `data/processed/`
- Processing time: ~35 seconds (first run), <1 second (cached)

### Documentation
- Added ITERATION_1_SUMMARY.md
- Added IMPROVEMENTS_SUMMARY.md
- Created initial README.md
- Added PROJECT_DESCRIPTION.md

---

## [0.1.0] - 2025-10-15 - Initial Setup

### Added
- Basic project structure
- Docker configuration (Dockerfile, docker-compose.yml)
- Requirements file (requirements.txt)
- Helper scripts (setup.sh, scripts/clean.sh)
- Git configuration (.gitignore)

### Documentation
- Created PROJECT_DESCRIPTION.md outline
- Created README.md skeleton

---

## Future Enhancements (Not Yet Implemented)

### Potential Improvements
- Hybrid search (dense + BM25 sparse retrieval)
- Reranking with cross-encoders
- Streamlit UI for interactive queries
- Query expansion and reformulation
- Multi-hop retrieval for complex questions
- Document summarization

### Testing & CI/CD
- Unit tests (pytest)
- Integration tests
- GitHub Actions CI pipeline
- Automated Docker builds

### Production Features
- Authentication and authorization
- Rate limiting
- Query result caching (Redis)
- Monitoring (Prometheus + Grafana)
- Distributed deployment (Kubernetes)

---

## Version History Summary

| Version | Date | Milestone |
|---------|------|-----------|
| 1.0.0 | 2025-10-21 | ✅ All 5 iterations complete, production-ready |
| 0.5.0 | 2025-10-20 | Iteration 5: Evaluation framework |
| 0.4.0 | 2025-10-19 | Iteration 4: Answer generation & Docker |
| 0.3.0 | 2025-10-18 | Iteration 3: Query & retrieval |
| 0.2.0 | 2025-10-17 | Iteration 2: Chunking & indexing |
| 0.1.0 | 2025-10-16 | Iteration 1: Foundation & preprocessing |
| 0.0.1 | 2025-10-15 | Initial setup |

---

## Notes

- All dates are approximate based on iteration completion
- Each iteration includes comprehensive documentation (see this CHANGELOG)
- See [ARCHITECTURE.md](ARCHITECTURE.md) for complete technical documentation
- See [DESIGN.md](DESIGN.md) for architectural decisions and trade-offs
- See [README.md](README.md) for quick start and usage guide

---

**Maintained by:** Mikkel Nielsen  
**Repository:** https://github.com/Medesen/portfolio/rag_pipeline  
**License:** MIT (see LICENSE file)

