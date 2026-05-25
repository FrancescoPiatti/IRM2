# src/utils/logger.py
import logging
import os
from datetime import datetime
from typing import Optional


class SimpleLogger:
    """
    Simple logger that writes logs to both console and a file.

    Notes
    -----
    - Avoids duplicate handlers across multiple instantiations in the same process.
    - Ensures each run can have a unique logger name (recommended in training loops).
    - Provides .close() to release file handles cleanly.
    """

    def __init__(
        self,
        name: str,
        log_dir: str,
        log_filename: str = "training.log",
        level: int = logging.INFO,
        *,
        add_console: bool = False,
        unique: bool = True,
        formatter: Optional[logging.Formatter] = None,
    ):
        self.base_name = str(name)
        self.log_dir = str(log_dir)

        if log_dir is None:
            raise ValueError("SimpleLogger: log_dir cannot be None (provide a real directory).")
        
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, log_filename)

        # Make logger name unique to prevent collisions across runs
        if unique:
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            self.name = f"{self.base_name}.{stamp}"
        else:
            self.name = self.base_name

        self._logger = self._configure_logger(
            level=level,
            add_console=add_console,
            formatter=formatter,
        )

    def _configure_logger(
        self,
        *,
        level: int,
        add_console: bool,
        formatter: Optional[logging.Formatter],
    ) -> logging.Logger:
        """
        Configure the logger with file + optional console handlers.
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(level)
        logger.propagate = False  # don't double-log via root logger

        if formatter is None:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        # ---------- File handler (avoid duplicates for same file) ----------
        if not self._has_file_handler(logger, self.log_path):
            fh = logging.FileHandler(self.log_path)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # ---------- Console handler ----------
        if add_console and not self._has_stream_handler(logger):
            sh = logging.StreamHandler()
            sh.setLevel(level)
            sh.setFormatter(formatter)
            logger.addHandler(sh)

        return logger

    @staticmethod
    def _has_file_handler(logger: logging.Logger, log_path: str) -> bool:
        """
        True if logger already has a FileHandler pointing to log_path.
        """
        log_path = os.path.abspath(log_path)
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if os.path.abspath(h.baseFilename) == log_path:
                        return True
                except Exception:
                    # defensive: some handlers may not expose baseFilename
                    continue
        return False

    @staticmethod
    def _has_stream_handler(logger: logging.Logger) -> bool:
        """
        True if logger already has a StreamHandler (console-like).
        """
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                return True
        return False

    def get_logger(self) -> logging.Logger:
        """
        Return the configured logger instance.
        """
        return self._logger

    def close(self) -> None:
        """
        Flush and close handlers to release resources cleanly.
        """
        handlers = list(self._logger.handlers)
        for h in handlers:
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
            try:
                self._logger.removeHandler(h)
            except Exception:
                pass