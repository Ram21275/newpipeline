"""Locked, restartable CUB final-evaluation workflow.

The package deliberately keeps protocol construction and statistical analysis
CPU-only.  GPU/model code is imported lazily so input audits and result
re-analysis work on a fresh machine without Transformers installed.
"""

from .core import SCHEMA_VERSION, canonical_hash

__all__ = ["SCHEMA_VERSION", "canonical_hash"]

