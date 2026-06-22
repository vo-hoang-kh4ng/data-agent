# The app-runtime `engine` pulls optional heavy/missing deps (sentence_transformers
# -> transformers/Keras + torch, plus `cache`/`ui` modules). On this Windows env that
# stack half-loads TensorFlow and later corrupts the heap (0xC0000374). The benchmark /
# eval path (TriadicDGMOrchestrator under benchmark/) does NOT need any of it, so by
# default we do NOT import engine at all. Set TRIADIC_LOAD_ENGINE=1 to opt back in.
import os

TriadicAgent = None  # type: ignore[assignment]
if os.environ.get("TRIADIC_LOAD_ENGINE", "0") in ("1", "true", "yes", "on"):
    try:
        from .engine import TriadicAgent  # noqa: F401
    except Exception:  # noqa: BLE001 - optional runtime path
        TriadicAgent = None

__all__ = ["TriadicAgent"]
