from __future__ import annotations

from pathlib import Path


def resolve_project_root(current_file: str | Path | None = None) -> Path:
    current = Path(current_file or __file__).resolve()
    for candidate in current.parents:
        if candidate.name == "team-builder":
            return candidate
    return current.parent


PROJECT_ROOT = resolve_project_root(__file__)
DATA_DIR = PROJECT_ROOT / "data"
