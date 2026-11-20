"""Runtime settings. Environment variables use the BRIEFWRIGHT_ prefix."""

from pydantic_settings import BaseSettings, SettingsConfigDict

# A real SciFact dev claim. The retriever has a stored MiniLM vector for this text.
DEFAULT_QUERY = "Birth-weight is positively associated with breast cancer."


class Settings(BaseSettings):
    """Process settings. Per-run knobs such as auto-approve live on the graph state."""

    model_config = SettingsConfigDict(
        env_prefix="BRIEFWRIGHT_",
        env_file=".env",
        extra="ignore",
    )

    llm_backend: str = "mock"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    max_revisions: int = 2
    max_research_rounds: int = 2
    score_threshold: float = 0.75
    request_timeout_s: float = 60.0
    python_timeout_s: float = 2.0
    memory_path: str = "data/memory.json"
    corpus_dir: str = ""
    auto_approve: bool = True
    log_level: str = "INFO"
