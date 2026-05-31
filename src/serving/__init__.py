# src/serving/__init__.py
from .app import app
from .drift_monitor import drift_monitor_loop

__all__ = ["app", "drift_monitor_loop"]
__version__ = "1.0.0"