"""
aria/core/parsers.py

Parsing utilities for MCP tool results.

WHY THIS FILE EXISTS:
Databricks UC Functions return results as a JSON envelope:
    {"format": "CSV", "value": "col1,col2\nval1,val2\n", "truncated": false}

The actual data is a CSV string inside the "value" key.
This is not obvious from the documentation — we discovered it
by logging the raw tool result during development.

This parser handles that format so every agent can call
parse_tool_result() without knowing the envelope structure.

LESSON LEARNED (important for interviews):
Never assume MCP tool output format. Always log the raw result
first, inspect it, then write a defensive parser.
Silent failures from wrong parsing are harder to debug than
explicit errors.
"""

import csv
import io
import json
from typing import Any


def parse_tool_result(raw_result: str) -> list[dict]:
    """
    Parse MCP tool result from Databricks UC Functions.

    Args:
        raw_result: Raw string returned by UCFunctionToolkit tool call.
                   Format: JSON envelope with CSV data inside.

    Returns:
        List of dicts, one per row. Empty list if parsing fails.

    Example input:
        '{"format": "CSV", "value": "col1,col2\\nval1,val2\\n", "truncated": false}'

    Example output:
        [{"col1": "val1", "col2": "val2"}]
    """
    if not raw_result:
        return []

    try:
        # Step 1: Parse outer JSON envelope
        envelope = json.loads(raw_result)

        # Step 2: Extract CSV string from "value" key
        csv_string = envelope.get("value", "")
        if not csv_string.strip():
            return []

        # Step 3: Parse CSV into list of dicts
        reader = csv.DictReader(io.StringIO(csv_string))
        records = []
        for row in reader:
            record = dict(row)
            # Convert numeric fields from string to float
            for field in ["amount", "min_premium", "max_coverage",
                          "rate_per_mille", "premium_paid", "sum_insured",
                          "claim_amount"]:
                if field in record and record[field]:
                    try:
                        record[field] = float(record[field])
                    except (ValueError, TypeError):
                        record[field] = 0.0
            records.append(record)

        return records

    except json.JSONDecodeError:
        # Some tools return plain CSV without envelope
        try:
            reader = csv.DictReader(io.StringIO(raw_result))
            return [dict(row) for row in reader]
        except Exception:
            return []

    except Exception:
        return []


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert any value to float.
    Used throughout agents when reading from parsed tool results.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert any value to int.
    """
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default
