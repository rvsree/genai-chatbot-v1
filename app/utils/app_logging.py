import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.config.app_config import AppConfig

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

def get_logger(cfg: AppConfig) -> logging.Logger:
    log = logging.getLogger("gl_rag_app")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)

    fh = RotatingFileHandler(Path(cfg.data_dir, "app.log"), maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    sh = logging.StreamHandler()

    fmt = logging.Formatter(LOG_FORMAT)
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(sh)
    return log
