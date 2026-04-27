import json
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger('HoradricConfig')
DEFAULT_CONFIG_PATH = Path.cwd() / 'horadric_conf.json'


@dataclass
class HoradricConfig:
    """Core directory layout and settings for all Horadric-powered projects."""

    # Root Directories
    data_dir: str = 'data'
    runtime_dir: str = 'runtime'

    # Global Settings
    cache_expiry_days: int = 180

    @classmethod
    def load_or_create(cls, config_path: Path | str = DEFAULT_CONFIG_PATH) -> 'HoradricConfig':
        path = Path(config_path)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
                return cls(**{k: v for k, v in data.items() if k in valid_keys})
            except json.JSONDecodeError:
                logger.error('Config file corrupted. Using defaults.', path=str(path))
                return cls()

        instance = cls()
        instance.save(path)
        return instance

    def save(self, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
