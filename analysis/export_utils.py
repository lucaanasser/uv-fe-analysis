"""Export utilities (CSV / JSON)."""
from __future__ import annotations
import json
import os
import pandas as pd
from typing import Iterable, Any


def ensure_dir(path: str) -> None:
    """Ensure directory exists (create if needed)."""
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Any, *, indent: int = 2) -> None:
    """Write JSON with UTF-8 encoding and indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def write_csv(path: str, df: pd.DataFrame) -> None:
    """Write DataFrame to CSV (UTF-8)."""
    df.to_csv(path, index=False)


def dataclasses_to_dataframe(items: Iterable[Any]) -> pd.DataFrame:
    """Convert a sequence of dataclass instances to a DataFrame."""
    rows = [getattr(x, "to_dict")() if hasattr(x, "to_dict") else x for x in items]
    return pd.DataFrame(rows)
