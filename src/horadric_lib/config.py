import json
from dataclasses import dataclass, asdict
from pathlib import Path
import structlog

logger = structlog.get_logger("HoradricConfig")

# Default to the directory where the script is executed
DEFAULT_CONFIG_PATH = Path.cwd() / "horadric_conf.json"


@dataclass
class HoradricConfig:
    """Core configuration for all Horadric Library utilities."""

    data_dir: str = "data"
    log_dir: str = "runtime/logs"
    registry_file: str = "data/dataset_registry.json"
    default_hf_source: str = "LLM4Code/SATBench"
    cache_expiry_days: int = 180

    @classmethod
    def load_or_create(
        cls, config_path: Path | str = DEFAULT_CONFIG_PATH
    ) -> "HoradricConfig":
        """Loads existing config, or creates a new one with defaults if missing."""
        path = Path(config_path)

        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
                filtered_data = {k: v for k, v in data.items() if k in valid_keys}
                return cls(**filtered_data)
            except json.JSONDecodeError:
                logger.error("Config file corrupted. Using defaults.", path=str(path))
                return cls()

        instance = cls()
        instance.save(path)
        logger.info("Created new configuration file", path=str(path))
        return instance

    def save(self, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        """Saves the current state to the JSON file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
