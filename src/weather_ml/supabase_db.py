from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.schema import MetaData, Table


def make_engine(db_url: str) -> Engine:
    return create_engine(db_url, pool_pre_ping=True)


def upsert_dataframe(
    engine: Engine,
    df: pd.DataFrame,
    *,
    table_name: str,
    conflict_columns: list[str],
) -> int:
    if df.empty:
        return 0

    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    statement = insert(table).values(records)
    update_columns = {
        column.name: statement.excluded[column.name]
        for column in table.columns
        if column.name not in {"id", *conflict_columns, "inserted_at", "generated_at"}
    }
    statement = statement.on_conflict_do_update(
        index_elements=conflict_columns,
        set_=update_columns,
    )

    with engine.begin() as connection:
        result = connection.execute(statement)
    return result.rowcount or 0


def assert_connection(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("select 1"))

