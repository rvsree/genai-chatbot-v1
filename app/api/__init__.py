# package marker for app.api
# Re-export the FastAPI `app` for convenience and to satisfy imports.
from .main import app  # re-export the FastAPI instance

__all__ = ["app"]
