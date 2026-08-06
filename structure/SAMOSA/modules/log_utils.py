from __future__ import annotations

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import logging

from enum import Enum
from multiprocessing import Queue

from logging.handlers import QueueHandler, QueueListener


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


LogLevelType = Literal[
    LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL
]


def main_logging_setup(log_queue: Queue, log_level: LogLevelType) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    s_handler = logging.StreamHandler()
    s_handler.setFormatter(get_formatter())
    root_logger.addHandler(s_handler)

    f_handler = logging.FileHandler(filename="samosa.log", mode="a")
    f_handler.setFormatter(get_formatter())
    root_logger.addHandler(f_handler)

    listener = QueueListener(log_queue, s_handler)
    listener.start()
    return root_logger, listener


def child_logging_setup(
    log_queue: Queue, process_name: str, log_level: LogLevelType
) -> logging.Logger:
    logger = logging.getLogger(process_name)
    logger.propagate = False
    logger.setLevel(log_level)

    queue_handler = QueueHandler(log_queue)
    if not any(isinstance(h, QueueHandler) for h in logger.handlers):
        logger.addHandler(queue_handler)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(filename="samosa.log", mode="a")
        file_handler.setFormatter(get_formatter())
        logger.addHandler(file_handler)
    return logger


def get_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s (%(funcName)s) | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name) -> logging.Logger:
    return logging.getLogger(name)

