import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path

import structlog


class ConsoleNoiseFilter(logging.Filter):
    """Filters out DEBUG and INFO logs from specific noisy third-party loggers."""

    NOISY_LOGGERS = (
        "openai",
        "httpx",
        "urllib3",
        "boto3",
        "paramiko",
        "httpcore",
        "binpickle",
    )

    def filter(self, record) -> bool:
        if record.name.startswith(self.NOISY_LOGGERS):
            return record.levelno >= logging.WARNING
        return True


def configure_logging(log_dir: str | Path, app_name: str | None = None) -> str:
    """
    Configures structlog to output:
    1. Pretty colorful text to Console.
    2. Structured JSON to a File in the specified log_dir.
    """

    if app_name is None:
        if sys.argv and sys.argv[0]:
            app_name = Path(sys.argv[0]).stem
        else:
            app_name = "app"
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile_path = str(log_dir_path / f"{app_name}_run_{timestamp}.jsonl")

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "console_noise_filter": {
                "()": ConsoleNoiseFilter,
            }
        },
        "formatters": {
            "json_file": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
                "foreign_pre_chain": shared_processors,
            },
            "colored_console": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(colors=True),
                "foreign_pre_chain": shared_processors,
            },
        },
        "handlers": {
            "console": {
                "level": "DEBUG",
                "class": "logging.StreamHandler",
                "formatter": "colored_console",
                "filters": ["console_noise_filter"],
            },
            "file": {
                "level": "DEBUG",
                "class": "logging.FileHandler",
                "filename": logfile_path,
                "formatter": "json_file",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": True,
            },
        },
    }

    logging.config.dictConfig(logging_config)
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return logfile_path
