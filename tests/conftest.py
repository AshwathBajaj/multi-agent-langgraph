import pytest

from briefwright.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        llm_backend="mock",
        memory_path=str(tmp_path / "memory.json"),
        auto_approve=True,
        max_revisions=2,
        log_level="WARNING",
    )
