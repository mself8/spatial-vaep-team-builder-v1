"""Shared parsing helpers used across the Team-Builder pipeline.

Consolidates `_safe_literal` and `_to_int` implementations that were
previously duplicated across multiple scripts.
"""
from __future__ import annotations

import ast

import pandas as pd


def _safe_literal(value, default=None):
    """Safely convert list/dict-like strings into Python objects.

    Returns ``default`` for NaN, empty strings, or parse failures.
    Pass ``default=[]`` when callers expect an iterable on failure
    (e.g. Wyscout positions/tags fields).
    """
    if isinstance(value, (list, dict)):
        return value
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return default
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return default


def _to_int(value, default=None):
    """Coerce a value to int, returning ``default`` on failure."""
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default
