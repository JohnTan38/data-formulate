def get_app():
    from .app import app
    return app

__all__ = ["get_app"]

# Avoid importing app at package import time (prevents duplicate registrations / circular imports).
# Import the app lazily in scripts that need it