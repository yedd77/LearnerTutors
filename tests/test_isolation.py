"""Guards that the suite can never read or write the developer's real runtime state."""
import tempfile
from pathlib import Path

import app.api.admin as admin_module
import app.config as config_module
import app.services.illustrations as illustrations
import app.services.vector_store as vs
from app.config import ROOT, get_settings

REAL_STORAGE = (ROOT / "storage").resolve()
TEMP_ROOT = Path(tempfile.gettempdir()).resolve()


def inside(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(parent)
        return True
    except ValueError:
        return False


def test_chroma_lives_in_a_temp_directory_not_real_storage():
    chroma = Path(get_settings().chroma_absolute_path)
    assert not inside(chroma, REAL_STORAGE)
    assert inside(chroma, TEMP_ROOT)
    assert vs.get_collection().name.startswith("test_chunks")


def test_runtime_config_is_redirected_to_a_per_test_file():
    assert not inside(config_module.RUNTIME_CONFIG_PATH, REAL_STORAGE)
    assert config_module.RUNTIME_CONFIG_PATH == admin_module.RUNTIME_CONFIG_PATH
    assert not config_module.RUNTIME_CONFIG_PATH.exists()


def test_illustrations_are_redirected():
    assert not inside(illustrations.IMAGE_DIR, REAL_STORAGE)


def test_real_provider_credentials_are_not_visible():
    settings = get_settings()
    assert settings.openai_api_key == "" and settings.gemini_api_key == ""
    assert settings.llm_provider == "" and settings.llm_model == ""
    assert settings.rag_api_key.startswith("test-api-key-")
