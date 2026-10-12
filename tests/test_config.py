import pytest

from config import Config


def test_project_root_is_the_repo_root():
    # Regression: it used to point one folder above the repo
    assert (Config.PROJECT_ROOT / "config.py").exists()


def test_pdf_dir_lives_inside_the_project():
    assert Config.PDF_DIR.is_relative_to(Config.PROJECT_ROOT)


def test_chunk_overlap_is_smaller_than_chunk_size():
    assert 0 <= Config.CHUNK_OVERLAP < Config.CHUNK_SIZE


def test_search_k_is_positive():
    assert Config.SEARCH_K > 0


def test_validate_requires_api_key(monkeypatch):
    monkeypatch.setattr(Config, "OPENAI_API_KEY", "")
    with pytest.raises(ValueError):
        Config.validate()
