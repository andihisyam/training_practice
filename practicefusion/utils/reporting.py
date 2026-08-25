from __future__ import annotations

from typing import Any


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def dict_table_rows(rows: list[dict[str, Any]], ordered_keys: list[str]) -> str:
    header = "| " + " | ".join(ordered_keys) + " |"
    sep = "| " + " | ".join(["---"] * len(ordered_keys)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for key in ordered_keys) + " |")
    return "\n".join([header, sep, *body])

