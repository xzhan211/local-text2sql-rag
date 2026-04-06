from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to repo root (two levels up from this file: app/core/config.py → repo root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    All configuration for the system in one place.

    Values are loaded from the environment or a .env file at the repo root.
    Every field has a sensible default so the system runs without a .env
    (except ANTHROPIC_API_KEY, which is required for LLM calls).

    Why pydantic-settings?
      - Type-safe: settings.top_k_examples is always an int, never a str
      - Validated at startup: bad config fails immediately, not mid-request
      - IDE-friendly: autocomplete on settings.*
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env vars (safe for shared environments)
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0          # deterministic for first SQL attempt
    llm_retry_temperature: float = 0.3    # slight variation on lesson-augmented retry

    # ── Embeddings ───────────────────────────────────────────────────────────
    # "all-MiniLM-L6-v2" is a good default: 384-dim, fast, free, runs locally.
    # Upgrade to "all-mpnet-base-v2" (768-dim) for better accuracy at higher cost.
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_examples: int = 5    # how many NLQ+SQL examples to include in the prompt
    top_k_lessons: int = 3     # how many lessons to retrieve on retry

    # Confidence is the average cosine similarity of the top-k retrieved examples.
    # Below this threshold, the system triggers a lesson-augmented retry.
    # 0.75 is a reasonable starting point; tune based on your data.
    confidence_threshold: float = 0.75

    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir: Path = BASE_DIR / "data"
    index_dir: Path = BASE_DIR / "data" / "indexes"   # FAISS index files live here
    sqlite_path: Path = BASE_DIR / "data" / "app.db"
    duckdb_path: Path = BASE_DIR / "data" / "sample_db.duckdb"

    # ── Training ─────────────────────────────────────────────────────────────
    eval_split: float = 0.2    # 20% held out for evaluation
    random_seed: int = 42


# Module-level singleton — import this everywhere:
#   from app.core.config import settings
settings = Settings()
