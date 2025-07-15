# convenience re‑exports
from importlib import import_module as _imp

__all__ = ["tsp", "gcolor", "promptopt", "codegen", "symboreg_oscillator1", "symboreg_oscillator2"]
for _m in __all__:
    globals()[_m] = _imp(f"tasks.{_m}")