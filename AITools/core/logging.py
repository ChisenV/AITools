import datetime
import inspect
import logging
import os
import sys

from enum import IntEnum
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from pathlib import Path
from queue import Queue
from types import FrameType
from typing import Optional, Dict, Any, List, Union
from concurrent.futures import ThreadPoolExecutor

_LOG_FILE_MAX_BYTES = 1024 * 1024 * 2  # 2MB
_LOG_FILE_BACKUP_COUNT = 5  # A maximum of five log files can be reserved


class LogLevel(IntEnum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def default_format(json_format: bool = False):
    return (
        '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", '
        '"module": "%(module)s", "message": "%(message)s"}' if json_format else
        "[%(asctime)s.%(msecs)03d][%(process)d][%(thread)d][%(levelname)s] "
        "%(name)s %(module)s L%(lineno)d: %(message)s"
    )


class LogManager:
    """
    Plugin log management system

    """
    NOTSET = LogLevel.NOTSET
    DEBUG = LogLevel.DEBUG
    INFO = LogLevel.INFO
    WARNING = LogLevel.WARNING
    WARN = WARNING
    ERROR = LogLevel.ERROR
    CRITICAL = LogLevel.CRITICAL
    FATAL = CRITICAL

    def __init__(
            self,
            name: str = "app",
            log_level: Union[int, str] = logging.INFO,
            handlers: Optional[List[logging.Handler]] = None,
            log_format: Optional[str] = None,
            log_dir: Optional[str] = None,
            json_format: bool = False,
            default_stream_handler: bool = True,
            default_file_handler: bool = True
    ):
        """
        Plugin log management system

        Args:
            name: Logger name
            log_level: Logger level
            handlers: Custom handler list
            log_format: Custom format string
            json_format: Whether to use JSON format
            default_stream_handler: Whether to use the default console output handler
            default_file_handler: Whether to use the default file output handler
        """
        self.log_level = log_level
        self.log_dir = log_dir
        self.json_format = json_format
        self.default_stream_handler = default_stream_handler
        self.default_file_handler = default_file_handler
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        if not self.logger.handlers:
            self._setup_initial_handlers(handlers, log_format)

    def _setup_initial_handlers(self, handlers, log_format):
        self.log_format = log_format or default_format(self.json_format)
        self.datefmt = '%Y-%m-%d %H:%M:%S'
        self.formatter = logging.Formatter(self.log_format, datefmt=self.datefmt)

        self.handlers: List[logging.Handler] = handlers or []
        self._init_default_handlers()

        for handler in self.handlers:
            handler.setFormatter(self.formatter)
            self.logger.addHandler(handler)

    def _init_default_handlers(self):
        """Initializes the default log handlers"""
        if self.default_stream_handler and not any(isinstance(h, logging.StreamHandler) for h in self.handlers):
            self.handlers.append(logging.StreamHandler(sys.stdout))

        # Add a file processor with automatic rotation
        if self.default_file_handler and not any(isinstance(h, logging.FileHandler) for h in self.handlers):
            logs_dir = self.log_dir if self.log_dir and Path(self.log_dir).is_dir() else os.path.join(os.getcwd(), 'logs')
            os.makedirs(logs_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(logs_dir, f"{timestamp}.log")

            file_handler = RotatingFileHandler(
                filename,
                maxBytes=_LOG_FILE_MAX_BYTES,  # 2MB
                backupCount=_LOG_FILE_BACKUP_COUNT,
                encoding='utf-8'
            )
            self.handlers.append(file_handler)

    def add_handler(self, handler: logging.Handler):
        """Add a custom processor"""
        handler.setFormatter(self.formatter)
        self.handlers.append(handler)
        self.logger.addHandler(handler)

    def log(
            self,
            level: Union[int, str],
            msg: str,
            frame: Optional[FrameType] = None,
            extra: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        """
        Logging method

        Args:
        level: Logging Level
        msg: Log Message
        extra: Extra data dictionary
        **kwargs: Other parameters (for different handlers)
        """
        if isinstance(level, str):
            level = logging.getLevelName(level.upper())
        if frame is None:
            frame = inspect.currentframe().f_back
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn=frame.f_code.co_filename,
            lno=frame.f_lineno,
            msg=msg,
            args=(),
            exc_info=None,
            func=frame.f_code.co_name,
            extra=extra,
            sinfo=None
        )

        # Distribute to all handlers
        for handler in self.handlers:
            handler.handle(record)

    def debug(self, msg, **kwargs):
        self.log(logging.DEBUG, msg, frame=inspect.currentframe().f_back, **kwargs)

    def info(self, msg, **kwargs):
        self.log(logging.INFO, msg, frame=inspect.currentframe().f_back, **kwargs)

    def warning(self, msg, **kwargs):
        self.log(logging.WARNING, msg, frame=inspect.currentframe().f_back, **kwargs)

    def error(self, msg, **kwargs):
        self.log(logging.ERROR, msg, frame=inspect.currentframe().f_back, **kwargs)

    def critical(self, msg, **kwargs):
        self.log(logging.CRITICAL, msg, frame=inspect.currentframe().f_back, **kwargs)

    def close(self):
        """Shut down all handlers"""
        print("close")
        for handler in self.handlers:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LogBridgeHandler(logging.Handler):
    def __init__(self, loggerWrap):
        """
        Log bridge handler

        :param loggerWrap: Encapsulated log objects
        """
        super().__init__()
        self.loggerWrap = loggerWrap

    def emit(self, record):
        """Send logs to the other system"""
        try:
            message = self.format(record)
            self.loggerWrap.log(record.levelno, message)
        except Exception as e:
            # logger.log(logging.ERROR, f"Error sending log to loggerWrap: {str(e)}")
            print(f"Error sending log to loggerWrap: {str(e)}")


class AsyncLogBridgeHandler(LogBridgeHandler):
    def __init__(self, loggerWrap, max_workers=2):
        super().__init__(loggerWrap)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def emit(self, record):
        """使用线程池异步处理日志"""
        try:
            self.executor.submit(self._async_emit, record)
        except Exception as e:
            print(f"Error submitting log task: {str(e)}")

    def _async_emit(self, record):
        """实际日志发送逻辑"""
        message = self.format(record)
        self.loggerWrap.log(record.levelno, message)

    def close(self):
        self.executor.shutdown(wait=True)


class AsyncLogManager(LogManager):
    def __init__(self, *args, queue_maxsize: int = 5000, **kwargs):
        self.queue_maxsize = queue_maxsize
        super().__init__(*args, **kwargs)
        self._setup_async_logging()

    def _setup_async_logging(self):
        self.log_queue = Queue(maxsize=self.queue_maxsize)

        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(self.formatter)

        self.original_handlers = self.handlers
        self.listener = QueueListener(
            self.log_queue, *self.original_handlers, respect_handler_level=True
        )
        self.listener.start()

        self.logger.handlers = [queue_handler]

    def close(self):
        """Ensure asynchronous closure of the process"""
        super().close()
        if hasattr(self, 'listener'):
            self.listener.stop()


def get_logger(name=__name__, handlers=None):
    logger = LogManager(
        name=name,
        log_level=LogLevel.INFO,
        handlers=handlers,
        default_file_handler=False
    )
    return logger
