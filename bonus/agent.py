"""bonus/agent.py — Hybrid Memory Agent POC.

Combines:
  1. Episodic Memory (Vector Store — Qdrant in-memory with fastembed)
  2. Stable User Profile & Activity (Feature Store — Feast online store)

Meets Bonus Challenge requirements:
  - remember(text, user_id): Chunks, embeds, and stores episodic memories with payload filtering.
  - recall(query, user_id): Fetches Feast user profile & velocity, searches Qdrant episodic memory,
    and returns assembled context.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from feast import FeatureStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


class HybridMemoryAgent:
    """Personal AI Assistant Agent with Hybrid Memory Architecture."""

    def __init__(
        self,
        feast_repo_path: str | Path | None = None,
        collection_name: str = "episodic_memory",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        # Resolve feast repo path
        if feast_repo_path is None:
            feast_repo_path = Path(__file__).resolve().parent.parent / "app" / "feast_repo"
        self.feast_repo_path = Path(feast_repo_path)

        # 1. Initialize Feature Store (Feast)
        self.fs = FeatureStore(repo_path=str(self.feast_repo_path))

        # 2. Initialize Vector Embedder & Vector Store (Qdrant)
        self.embedder = TextEmbedding(model_name=embedding_model)
        self.vector_dim = 384  # bge-small-en-v1.5 produces 384-dimensional vectors
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:")

        # Create Qdrant collection for episodic memory
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
        )
        self._point_counter = 0

    def _chunk_text(self, text: str, max_chunk_chars: int = 300) -> list[str]:
        """Simple paragraph/sentence-boundary aware chunking."""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks: list[str] = []
        for p in paragraphs:
            if len(p) <= max_chunk_chars:
                chunks.append(p)
            else:
                # Split on sentence boundaries
                sentences = p.replace(". ", ".\n").split("\n")
                buf = ""
                for s in sentences:
                    if len(buf) + len(s) < max_chunk_chars:
                        buf = f"{buf} {s}".strip()
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = s
                if buf:
                    chunks.append(buf)
        return chunks if chunks else [text]

    def remember(self, text: str, user_id: str = "u_001", metadata: dict[str, Any] | None = None) -> list[str]:
        """Add a new piece of episodic memory for this user.

        1. Chunks text into coherent knowledge units.
        2. Computes dense vector embeddings.
        3. Upserts to Qdrant collection with user_id payload for multi-tenant isolation.
        """
        chunks = self._chunk_text(text)
        vectors = list(self.embedder.embed(chunks))
        points: list[PointStruct] = []
        memory_ids: list[str] = []
        now_ts = time.time()

        for chunk, vec in zip(chunks, vectors):
            self._point_counter += 1
            mem_id = str(uuid.uuid4())[:8]
            memory_ids.append(mem_id)

            payload = {
                "memory_id": mem_id,
                "user_id": user_id,
                "text": chunk,
                "created_at": now_ts,
            }
            if metadata:
                payload.update(metadata)

            points.append(
                PointStruct(
                    id=self._point_counter,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return memory_ids

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve top-K memories + user profile features -> return assembled context.

        1. Fetches stable user profile and real-time velocity from Feast online store.
        2. Runs vector similarity search in Qdrant with payload filter by user_id.
        3. Synthesizes an assembled prompt context for downstream LLM generation.
        """
        # --- A. Retrieve Features from Feast Online Store ---
        profile_features = {
            "reading_speed_wpm": 200,
            "preferred_language": "vi",
            "topic_affinity": "general",
            "queries_last_hour": 0,
            "distinct_topics_24h": 0,
        }
        try:
            feast_res = self.fs.get_online_features(
                features=[
                    "user_profile_features:reading_speed_wpm",
                    "user_profile_features:preferred_language",
                    "user_profile_features:topic_affinity",
                    "query_velocity_features:queries_last_hour",
                    "query_velocity_features:distinct_topics_24h",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()

            for k in ("reading_speed_wpm", "preferred_language", "topic_affinity", "queries_last_hour", "distinct_topics_24h"):
                val = feast_res.get(k)
                if val and len(val) > 0 and val[0] is not None:
                    profile_features[k] = val[0]
        except Exception as e:
            # Fallback for mock/resilience
            pass

        # --- B. Retrieve Episodic Memories from Qdrant Vector Store ---
        q_vec = next(self.embedder.embed([query])).tolist()

        # Enforce user_id tenant isolation filter
        user_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            query_filter=user_filter,
            limit=top_k,
        ).points

        # --- C. Assemble Final Context ---
        context_lines: list[str] = [
            f"=== ASSEMBLED CONTEXT FOR USER [{user_id}] ===",
            f"Query: {query!r}",
            "",
            "--- [Feast: User Stable Profile] ---",
            f"  • Preferred Language : {profile_features['preferred_language']}",
            f"  • Reading Speed      : {profile_features['reading_speed_wpm']} WPM",
            f"  • Topic Affinity     : {profile_features['topic_affinity']}",
            "",
            "--- [Feast: Recent Activity Velocity] ---",
            f"  • Queries (Last 1h)  : {profile_features['queries_last_hour']}",
            f"  • Distinct Topics 24h: {profile_features['distinct_topics_24h']}",
            "",
            "--- [Qdrant: Relevant Episodic Memories] ---",
        ]

        if hits:
            for idx, h in enumerate(hits, 1):
                mem_text = h.payload.get("text", "")
                score = h.score
                context_lines.append(f"  {idx}. [Score: {score:.3f}] {mem_text}")
        else:
            context_lines.append("  (No relevant episodic memories found)")

        context_lines.append("==============================================")
        return "\n".join(context_lines)
