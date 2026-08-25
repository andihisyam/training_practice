from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def load_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

