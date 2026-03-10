import importlib
import importlib.util
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_core_types_and_service_import() -> None:
    assert importlib.util.find_spec("legal_rag") is not None

    config_module = importlib.import_module("legal_rag.config")
    services_module = importlib.import_module("legal_rag.services")
    types_module = importlib.import_module("legal_rag.types")

    AppConfig = config_module.AppConfig
    LegalAssistantService = services_module.LegalAssistantService
    RouteDecision = types_module.RouteDecision

    config = AppConfig()
    service = LegalAssistantService(config=config)
    assert service is not None
    assert RouteDecision.__annotations__["selected_mode"] is str
