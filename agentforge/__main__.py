"""Module entry point so AgentForge can run as ``python -m agentforge``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":  # pragma: no cover - thin shim
    sys.exit(main())
