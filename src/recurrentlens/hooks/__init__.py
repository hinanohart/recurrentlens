"""Hook registry for recurrent-state-aware activation capture."""

from recurrentlens.hooks.registry import (
    HookHandle,
    HookManager,
    register_hook,
    resolve_target,
)

__all__ = ["HookHandle", "HookManager", "register_hook", "resolve_target"]
