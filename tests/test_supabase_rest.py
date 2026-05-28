from types import SimpleNamespace

import pytest

from weather_ml.supabase_rest import _raise_schema_cache_hint


def test_schema_cache_error_includes_table_setup_hint() -> None:
    with pytest.raises(RuntimeError) as error:
        _raise_schema_cache_hint(SimpleNamespace(code="PGRST205"), "KadikoyWeatherCodeRaw")

    message = str(error.value)
    assert "KadikoyWeatherCodeRaw" in message
    assert "sql/005_kadikoy_weather_code_tables.sql" in message
    assert "SUPABASE_RAW_TABLE" in message


def test_non_schema_cache_error_is_reraised() -> None:
    original = RuntimeError("different failure")

    with pytest.raises(RuntimeError) as error:
        _raise_schema_cache_hint(original, "KadikoyWeatherCodeRaw")

    assert error.value is original
