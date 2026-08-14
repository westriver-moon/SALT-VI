"""Compatibility import for the validator v3 package.

The former monolithic and value-strict implementation has been removed. New
code should import from :mod:`semantic_imagination.validator`.
"""

from .validator import validate_atomic_response

__all__ = ["validate_atomic_response"]
