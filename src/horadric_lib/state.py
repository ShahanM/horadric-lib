from pathlib import Path


class StateTracker:
    """Manages temporary runtime state, like OpenAI Batch IDs."""

    def __init__(self, runtime_dir: str | Path = 'runtime') -> None:
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def save_tracking_id(self, job_name: str, batch_id: str) -> None:
        path = self.runtime_dir / f'{job_name}_active_batch.txt'
        with open(path, 'w') as f:
            f.write(batch_id)

    def get_tracking_id(self, job_name: str) -> str | None:
        path = self.runtime_dir / f'{job_name}_active_batch.txt'
        if path.exists():
            with open(path) as f:
                return f.read().strip()
        return None

    def clear_tracking_id(self, job_name: str) -> None:
        path = self.runtime_dir / f'{job_name}_active_batch.txt'
        if path.exists():
            path.unlink()
