"""
Two-index knowledge base manager.

This is the primary interface used by the training and inference pipelines.
It owns both FAISS indexes and the embedder, and coordinates with SQLite
to maintain the faiss_index_id bridge.

Two indexes:
  Index 1 (kb_nlq_sql_pairs):  NLQ+SQL few-shot examples
  Index 2 (kb_lessons_learned): Lessons learned from failures

Key design pattern — add_example():
    1. Embed the NLQ
    2. Add the vector to FaissIndex → get faiss_id
    3. Call update_pair_faiss_id(db_row_id, faiss_id) to link SQLite ↔ FAISS
    4. Return faiss_id

Key design pattern — search_examples():
    1. Embed the NLQ
    2. Search FaissIndex → [(faiss_id, score), ...]
    3. For each faiss_id, fetch the SQLite row via get_pair_by_faiss_id
    4. Return list of dicts with {nlq, sql, score}

The separation between "add to FAISS" and "save to disk" is intentional.
The training pipeline adds many vectors in a loop and calls save_indexes()
once at the end — not once per vector.
"""

from app.core.config import settings
from app.db.models import Lesson
from app.db.sqlite_client import (
    get_lesson_by_faiss_id,
    get_pair_by_faiss_id,
    update_lesson_faiss_id,
    update_pair_faiss_id,
)
from app.embeddings.embedder import encode
from app.vectorstore.faiss_index import FaissIndex

# Disk paths for the two index files
_INDEX1_PATH = settings.index_dir / "index1_examples.bin"
_INDEX2_PATH = settings.index_dir / "index2_lessons.bin"


class KBManager:
    """
    Knowledge Base Manager.

    Instantiate once and share across the lifetime of a pipeline run.
    Call load_indexes() after init to restore persisted state from disk.

    Example (inference):
        kb = KBManager()
        kb.load_indexes()
        results = kb.search_examples("how many orders last month", k=5)

    Example (training):
        kb = KBManager()
        kb.load_indexes()
        for nlq, sql, row_id in pairs:
            kb.add_example(nlq, sql, row_id)
        kb.save_indexes()
    """

    def __init__(self) -> None:
        self._index1 = FaissIndex(dim=settings.embedding_dim)  # examples
        self._index2 = FaissIndex(dim=settings.embedding_dim)  # lessons

    # ── Index 1: NLQ+SQL examples ─────────────────────────────────────────────

    def add_example(self, nlq: str, sql: str, db_row_id: int) -> int:
        """
        Embed NLQ and add to Index 1. Updates SQLite faiss_index_id.

        Args:
            nlq:        Natural language question.
            sql:        Corresponding gold SQL.
            db_row_id:  The SQLite `nlq_sql_pairs.id` for this record.

        Returns:
            The FAISS ID assigned to this vector.
        """
        vec = encode(nlq)
        faiss_ids = self._index1.add(vec)
        faiss_id = faiss_ids[0]
        update_pair_faiss_id(db_row_id, faiss_id)
        return faiss_id

    def search_examples(self, nlq: str, k: int | None = None) -> list[dict]:
        """
        Retrieve the top-k most similar NLQ+SQL examples for a given query.

        Args:
            nlq: The natural language question to search for.
            k:   Number of results (defaults to settings.top_k_examples).

        Returns:
            List of dicts: [{nlq, sql, score}, ...], sorted by score descending.
            Score is cosine similarity in [-1.0, 1.0].
            Returns [] if index is empty.
        """
        k = k or settings.top_k_examples
        vec = encode(nlq)
        hits = self._index1.search(vec, k)

        results = []
        for faiss_id, score in hits:
            row = get_pair_by_faiss_id(faiss_id)
            if row is not None:
                results.append({"nlq": row["nlq"], "sql": row["sql"], "score": score})
        return results

    # ── Index 2: lessons ──────────────────────────────────────────────────────

    def add_lesson(self, lesson: Lesson, db_row_id: int) -> int:
        """
        Embed lesson.trigger and add to Index 2. Updates SQLite faiss_index_id.

        We embed the `trigger` field (not title/fix_rule) because trigger describes
        the NLQ pattern that activates this lesson — it's the right semantic anchor
        for matching against incoming queries.

        Args:
            lesson:     A Lesson object (from app.db.models).
            db_row_id:  The SQLite `lessons.id` for this record.

        Returns:
            The FAISS ID assigned to this vector.
        """
        vec = encode(lesson.trigger)
        faiss_ids = self._index2.add(vec)
        faiss_id = faiss_ids[0]
        update_lesson_faiss_id(db_row_id, faiss_id)
        return faiss_id

    def search_lessons(self, nlq: str, k: int | None = None) -> list[dict]:
        """
        Retrieve the top-k most relevant lessons for a given NLQ.

        Used during inference retry and evaluation retry.

        Args:
            nlq: The natural language question that failed on first attempt.
            k:   Number of results (defaults to settings.top_k_lessons).

        Returns:
            List of dicts with all lesson fields + score:
            [{title, trigger, diagnosis, fix_rule, error_category, example, id, score}, ...]
            Returns [] if index is empty.
        """
        k = k or settings.top_k_lessons
        vec = encode(nlq)
        hits = self._index2.search(vec, k)

        results = []
        for faiss_id, score in hits:
            row = get_lesson_by_faiss_id(faiss_id)
            if row is not None:
                results.append({
                    "id":             row["id"],
                    "title":          row["title"],
                    "trigger":        row["trigger"],
                    "diagnosis":      row["diagnosis"],
                    "fix_rule":       row["fix_rule"],
                    "error_category": row["error_category"],
                    "example":        row["example"],
                    "score":          score,
                })
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_indexes(self) -> None:
        """Persist both indexes to disk. Call once after a batch of adds."""
        self._index1.save(_INDEX1_PATH)
        self._index2.save(_INDEX2_PATH)

    def load_indexes(self) -> None:
        """
        Load both indexes from disk into memory.

        Safe to call even if index files don't exist yet (they'll stay empty).
        Call this at the start of every pipeline run to restore persisted state.
        """
        self._index1.load(_INDEX1_PATH)
        self._index2.load(_INDEX2_PATH)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def example_count(self) -> int:
        return self._index1.size

    @property
    def lesson_count(self) -> int:
        return self._index2.size
