"""Long-memory experiment primitives for SWE-bench.

This package intentionally sits beside, rather than inside, the upstream SWE-bench
inference pipeline.  It makes the continual-memory protocol explicit and keeps a
disabled (No Memory) arm free of memory I/O and model calls.
"""

from .models import MemoryMode, MemoryPacket, ReflectionCase, TaskRecord
from .runtime import LongMemoryRuntime, MemoryConfig

__all__ = [
    "LongMemoryRuntime",
    "MemoryConfig",
    "MemoryMode",
    "MemoryPacket",
    "ReflectionCase",
    "TaskRecord",
]
