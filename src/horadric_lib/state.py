import json
from pathlib import Path
from typing import Any


class StateTracker:
    """Manages ephemeral runtime state (Batch IDs, JSON Manifests, Checkpoints)."""

    def __init__(self, runtime_dir: str | Path = 'runtime') -> None:
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def save_text_state(self, job_name: str, data: str) -> None:
        path = self.runtime_dir / f'{job_name}_state.txt'
        with open(path, 'w') as f:
            f.write(data)

    def get_text_state(self, job_name: str) -> str | None:
        path = self.runtime_dir / f'{job_name}_state.txt'
        if path.exists():
            with open(path) as f:
                return f.read().strip()
        return None

    def save_json_state(self, job_name: str, data: dict[str, Any]) -> None:
        path = self.runtime_dir / f'{job_name}_manifest.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def get_json_state(self, job_name: str) -> dict[str, Any] | None:
        path = self.runtime_dir / f'{job_name}_manifest.json'
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
