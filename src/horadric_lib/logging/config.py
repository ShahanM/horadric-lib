import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path

import structlog
from structlog.typing import EventDict, WrappedLogger

try:
    from asgi_correlation_id import correlation_id

    HAS_CORRELATION_ID = True
except ImportError:
    correlation_id = None
    HAS_CORRELATION_ID = False


def add_correlation_id(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Injects the correlation ID if the web logging extra is installed."""
    if HAS_CORRELATION_ID and correlation_id is not None:
        req_id = correlation_id.get()
        if req_id:
            event_dict['request_id'] = req_id
    return event_dict


class ConsoleNoiseFilter(logging.Filter):
    """Filters out DEBUG and INFO logs from specific noisy third-party loggers."""

    NOISY_LOGGERS = (
        'openai',
        'httpx',
        'urllib3',
        'boto3',
        'paramiko',
        'httpcore',
        'binpickle',
    )

    def filter(self, record) -> bool:
        if record.name.startswith(self.NOISY_LOGGERS):
            return record.levelno >= logging.WARNING
        return True


def configure_logging(log_dir: str | Path, app_name: str | None = None) -> str | None:
    """Configures structlog to output.

    1. Pretty colorful text to Console.
    2. Structured JSON to a File in the specified log_dir.
    """
    is_lambda = 'AWS_LAMBDA_FUNCTION_NAME' in os.environ
    logfile_path = None
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_correlation_id,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    formatters = {
        'json_formatter': {
            '()': structlog.stdlib.ProcessorFormatter,
            'processor': structlog.processors.JSONRenderer(),
            'foreign_pre_chain': shared_processors,
        }
    }
    handlers = {}
    active_handlers = []

    if is_lambda:
        handlers['console'] = {
            'level': 'INFO' if 'AWS_LAMBDA_RUNTIME_API' in os.environ else 'DEBUG',
            'class': 'loggin.StreamHandler',
            'stream': sys.stdout,
            'formatter': 'json_formatter',
            'filters': ['console_noise_filter'],
        }
        active_handlers.append('console')
    else:
        if app_name is None:
            app_name = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else 'app'
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logfile_path = str(log_dir_path / f'{app_name}_run_{timestamp}.jsonl')

        formatters['colored_console'] = {
            '()': structlog.stdlib.ProcessorFormatter,
            'processor': structlog.dev.ConsoleRenderer(colors=True),
            'foreign_pre_chain': shared_processors,
        }
        handlers['console'] = {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'colored_console',
            'filters': ['console_noise_filter'],
        }
        handlers['file'] = {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': logfile_path,
            'formatter': 'json_formatter',
            'encoding': 'utf-8',
        }
        active_handlers.extend(['console', 'file'])

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'filters': {'console_noise_filter': {'()': ConsoleNoiseFilter}},
        'formatters': formatters,
        'handlers': handlers,
        'loggers': {
            '': {
                'handlers': active_handlers,
                'level': 'DEBUG',
                'propagate': True,
            },
        },
    }

    logging.config.dictConfig(logging_config)
    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return logfile_path
