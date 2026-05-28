from __future__ import annotations

import math
from typing import Any

import pandas as pd
from postgrest.exceptions import APIError
from supabase import create_client


def classify_supabase_key(key: str) -> str:
    if key.startswith("sb_publishable_"):
        return "publishable"
    if key.startswith("sb_secret_"):
        return "secret"
    if key.startswith("eyJ"):
        return "jwt"
    return "unknown"


def upsert_dataframe_rest(
    *,
    supabase_url: str,
    supabase_key: str,
    table_name: str,
    frame: pd.DataFrame,
    on_conflict: str,
    batch_size: int = 1000,
) -> int:
    if frame.empty:
        return 0

    client = create_client(supabase_url, supabase_key)
    records = _records_for_json(frame)
    written = 0

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        try:
            client.table(table_name).upsert(batch, on_conflict=on_conflict).execute()
        except APIError as exc:
            try:
                _raise_schema_cache_hint(exc, table_name)
            except RuntimeError as hint:
                raise hint from exc
        written += len(batch)

    return written


def _raise_schema_cache_hint(exc: APIError, table_name: str) -> None:
    if getattr(exc, "code", None) != "PGRST205":
        raise exc

    raise RuntimeError(
        "Supabase REST could not find table "
        f"{table_name!r} in the exposed schema cache. Apply "
        "sql/005_kadikoy_weather_code_tables.sql to the target Supabase project, "
        "then rerun the pipeline. If the table already exists, reload the PostgREST "
        "schema cache and verify SUPABASE_RAW_TABLE/SUPABASE_FEATURE_TABLE match the "
        "exact case-sensitive table names."
    )


def read_table_rest(
    *,
    supabase_url: str,
    supabase_key: str,
    table_name: str,
    order_by: str = "observed_at",
    batch_size: int = 1000,
    max_rows: int | None = None,
) -> pd.DataFrame:
    client = create_client(supabase_url, supabase_key)
    rows: list[dict[str, Any]] = []
    start = 0

    while True:
        end = start + batch_size - 1
        response = (
            client.table(table_name)
            .select("*")
            .order(order_by)
            .range(start, end)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        if max_rows is not None and len(rows) >= max_rows:
            rows = rows[:max_rows]
            break
        if len(batch) < batch_size:
            break
        start += batch_size

    return pd.DataFrame(rows)


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notnull(frame), None)
    records = clean.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if hasattr(value, "isoformat"):
                record[key] = value.isoformat()
            elif isinstance(value, float) and math.isnan(value):
                record[key] = None
    return records
