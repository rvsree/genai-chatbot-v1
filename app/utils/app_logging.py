# app/utils/app_logging.py
# Restored, previously-working rotating file + console logger using provided AppConfig.
# No behavior change; only ensures log directory exists.

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.config.app_config import AppConfig

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

def get_logger(cfg: AppConfig) -> logging.Logger:
    logger = logging.getLogger("gl_rag_app")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    # Ensure directories exist
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(cfg.log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# import logging
# from logging.handlers import RotatingFileHandler
# from pathlib import Path
# from app.config.app_config import AppConfig
#
# LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
#
# def get_logger(cfg: AppConfig) -> logging.Logger:
#     log = logging.getLogger("gl_rag_app")
#     if log.handlers:
#         return log
#     log.setLevel(logging.INFO)
#     Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
#
#     fh = RotatingFileHandler(Path(cfg.data_dir, "app.log"), maxBytes=2_000_000, backupCount=3, encoding="utf-8")
#     sh = logging.StreamHandler()
#
#     fmt = logging.Formatter(LOG_FORMAT)
#     fh.setFormatter(fmt)
#     sh.setFormatter(fmt)
#
#     log.addHandler(fh)
#     log.addHandler(sh)
#     return log
