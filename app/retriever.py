"""
Retrieval over the knowledge-base chunks.

Design choices (documented for the README):
- TF-IDF + cosine similarity instead of a hosted embedding API. Zero external
  cost/latency for indexing, fully deterministic, and good enough for this
  corpus size (53 chunks). Swappable for real embeddings later (see
  `EmbeddingRetriever` stub at bottom).
- Precedence: we boost `status: active` and `policy_authority: official`
  chunks, and downrank/never-cite `status: superseded` or `draft` /
  `policy_authority: none` chunks as authority. Superseded chunks can still be
  *returned* (so the agent can explain "this used to be the policy") but are
  flagged `is_authoritative=False` and the agent's system prompt is told never
  to cite them as current truth.
- Conflict surfacing: if two *active + official* chunks about the same topic
  disagree (e.g. two different day counts), we don't silently pick one -- we
  return both and let the agent prompt handle telling the user there's a
  conflict + recommending human help.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ingest import Chunk, load_knowledge_base


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    is_authoritative: bool


class Retriever:
    def __init__(self, kb_dir: str):
        self.chunks: list[Chunk] = load_knowledge_base(kb_dir)
        corpus = [f"{c.heading}\n{c.text}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    @staticmethod
    def _is_authoritative(meta: dict[str, Any]) -> bool:
        return (
            meta.get("status") == "active"
            and meta.get("policy_authority") == "official"
            and meta.get("audience", "customer") == "customer"
        )

    def retrieve(self, query: str, top_k: int = 6) -> list[RetrievedChunk]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]

        scored: list[RetrievedChunk] = []
        for chunk, sim in zip(self.chunks, sims):
            if sim <= 0:
                continue
            authoritative = self._is_authoritative(chunk.metadata)
            # Boost authoritative, customer-facing, active/official content.
            # Heavily penalize non-customer-answering / draft / internal docs
            # so they only surface if truly nothing else matches.
            adj = sim
            if authoritative:
                adj *= 1.35
            if chunk.metadata.get("customer_answering") == "false":
                adj *= 0.05
            if chunk.metadata.get("status") == "draft":
                adj *= 0.1
            if chunk.metadata.get("audience") == "internal":
                adj *= 0.2
            scored.append(RetrievedChunk(chunk=chunk, score=float(adj), is_authoritative=authoritative))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


# --- Optional upgrade path -------------------------------------------------
# class EmbeddingRetriever(Retriever):
#     """Swap TF-IDF for e.g. `text-embedding-3-small` or a local
#     sentence-transformers model if the corpus grows. Same interface."""
#     ...


if __name__ == "__main__":
    import os

    r = Retriever(os.path.join(os.path.dirname(__file__), "..", "knowledge-base"))
    for rc in r.retrieve("How long do I have to return a backpack?"):
        print(f"{rc.score:.3f} auth={rc.is_authoritative} {rc.chunk.source_label()}")
