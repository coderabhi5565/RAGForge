# RAGForge

> Adaptive RAG system with measurable retrieval and generation quality.

RAGForge is a production-oriented Retrieval-Augmented Generation (RAG) system designed to retrieve reliable information from documents, intelligently route queries, and provide grounded answers with source citations.

The main goal of RAGForge is not just to build a working RAG chatbot, but to **experimentally evaluate and improve the retrieval and generation pipeline using measurable metrics**.

---

## 🎯 Project Goals

RAGForge focuses on three areas:

1. **Adaptive Retrieval**
   - Route queries based on their information needs.
   - Retrieve information from uploaded documents.
   - Use general LLM knowledge when appropriate.
   - Use web search for queries requiring current information.

2. **Retrieval Engineering**
   - Experiment with different chunking strategies.
   - Compare dense and hybrid retrieval.
   - Implement query rewriting.
   - Evaluate re-ranking strategies.
   - Preserve document metadata for source attribution.

3. **Quantitative Evaluation**
   - Build a ground-truth evaluation dataset.
   - Measure retrieval quality using Hit@K, Recall@K and MRR.
   - Measure answer quality using faithfulness and correctness.
   - Track latency, token usage and cost.
   - Compare different RAG configurations using reproducible experiments.

---

## 🏗️ Planned Architecture

```text
                         User Query
                             │
                             ▼
                    ┌─────────────────┐
                    │  Query Router   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Document RAG     General LLM     Web Search
              │
              ▼
        Query Rewriting
              │
              ▼
      Retrieval Pipeline
              │
              ▼
        Candidate Chunks
              │
              ▼
          Re-ranking
              │
              ▼
       Relevant Context
              │
              ▼
             LLM
              │
              ▼
       Answer + Citations

An evaluation pipeline will run alongside the system:

                    Evaluation Engine
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
         Retrieval      Generation      System
          Metrics         Metrics       Metrics
             │             │             │
          Hit@K         Faithfulness    Latency
          Recall@K      Correctness     Token Usage
          MRR           Relevance       Cost
🔬 Evaluation Strategy

RAGForge will use a ground-truth question-answer dataset containing questions, expected answers, and source information.

The system will be evaluated across multiple configurations rather than relying on subjective judgments.

Retrieval Metrics
Hit@1
Hit@3
Hit@5
Recall@K
Mean Reciprocal Rank (MRR)
Generation Metrics
Faithfulness
Answer correctness
Context relevance
Citation accuracy
System Metrics
p50 latency
p95 latency
Token usage
Cost per query
Throughput where relevant

Quantitative results will be added after the experiments are implemented and executed. No benchmark numbers are claimed before measurement.

🧪 Planned Experiments

The system will establish a baseline and progressively evaluate retrieval improvements.

Chunking

Compare:

Fixed-size chunking
Different chunk sizes
Different overlap sizes
Semantic chunking
Retrieval

Compare:

Dense vector retrieval
Keyword retrieval
Hybrid retrieval
Query Processing

Evaluate:

Original query
Query rewriting
Re-ranking

Compare:

Baseline retrieval
Retrieval + re-ranking

Each experiment will be evaluated against the same benchmark dataset wherever possible.

📊 Experiment Methodology

Every major retrieval change will follow the same process:

Baseline
   ↓
Change one component
   ↓
Run evaluation dataset
   ↓
Measure metrics
   ↓
Compare results
   ↓
Accept / reject the change

The objective is to understand the quality–latency–cost trade-offs of different RAG techniques rather than adding components without measurable benefit.

🛠️ Technology Stack
Component	Technology
Language	Python
API	FastAPI
LLM	OpenAI
Embeddings	OpenAI Embeddings
Vector Database	Qdrant
Document Processing	PyPDF
Text Splitting	LangChain Text Splitters
Workflow Orchestration	LangGraph
Containerization	Docker
Evaluation	Custom Evaluation Harness

The final stack may change when experiments show that another approach is more appropriate.

📁 Project Structure
RAGForge/
│
├── app/
│   ├── api/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── models/
│   ├── config/
│   └── main.py
│
├── data/
├── tests/
├── scripts/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

The structure will evolve as the system grows.

🚧 Development Phases
Phase 1 — Baseline RAG
 Document ingestion
 PDF/TXT processing
 Page-aware metadata
 Baseline chunking
 Embedding generation
 Qdrant vector storage
 Similarity retrieval
 LLM generation
 Source citations
 FastAPI endpoints
Phase 2 — Evaluation Harness
 Ground-truth QA dataset
 Automated retrieval evaluation
 Hit@K
 Recall@K
 MRR
 Generation evaluation
 Latency tracking
 Token/cost tracking
Phase 3 — Retrieval Optimization
 Chunking experiments
 Semantic chunking
 Hybrid retrieval
 Query rewriting
 Re-ranking
 Experiment comparison
Phase 4 — Adaptive RAG
 Query classification
 Document route
 General knowledge route
 Web search route
 Conditional retrieval
 LangGraph orchestration
Phase 5 — Productionization
 Docker
 Structured logging
 Error handling
 Observability
 Performance optimization
 API hardening
 Production deployment
📈 Results

This section will contain measured results after the evaluation pipeline is implemented.

Example structure:

Configuration	Hit@5	MRR	Faithfulness	p95 Latency
Baseline	—	—	—	—
+ Improved Chunking	—	—	—	—
+ Hybrid Retrieval	—	—	—	—
+ Re-ranking	—	—	—	—

All values will be generated from actual experiments.

🎯 Engineering Principles

RAGForge follows a few core principles:

Measure before optimizing

No optimization is considered successful without measurable improvement.

Prefer simple baselines

Advanced components are introduced only when they solve a demonstrated problem.

Reproducible experiments

Experiments should use consistent datasets and evaluation procedures.

Preserve provenance

Retrieved information should retain enough metadata to trace answers back to their source.

Optimize trade-offs

Retrieval quality, latency and cost must be considered together.

🔐 Security

Sensitive configuration such as API keys will be stored in environment variables.

.env files and secrets must never be committed to the repository.

Additional security measures will be added as the system moves toward production.

📌 Project Status

Current Phase: Phase 1 — Baseline RAG Foundation

The project is currently under active development.
