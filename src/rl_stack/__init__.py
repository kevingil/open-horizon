"""RL stack package.

Layers:
- domain: pure models, contracts, events
- application: orchestration + event bus
- infrastructure: concrete adapters (policy, env, rewards, store, tools)
- interface: FastAPI + WebSocket
"""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
