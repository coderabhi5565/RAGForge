
from app.retrieval.query_rewriter import QueryRewriter

rewriter = QueryRewriter()

question = "How does BM25 improve retrieval in a RAG system?"

rewritten = rewriter.rewrite(question)

print("Original query:", question)
print("Rewritten query:", rewritten)
