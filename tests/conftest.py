"""Keep provider-backed tests opt-in; ordinary pytest never loads real credentials."""

from pathlib import Path

LIVE_MODULES = {"test_integration.py", "test_multi_agent_integration.py", "test_skill_execution.py"}


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run real provider integrations (credentials and possible charges required)",
    )


def pytest_ignore_collect(collection_path: Path, config):
    if collection_path.name in LIVE_MODULES and not config.getoption("--live"):
        return True
    return None
