from __future__ import annotations

import pandas as pd
import pytest

from ab_glm.security import SecurityError, hash_pii_columns, validate_file_path


def test_validate_file_path_allows_valid_file() -> None:
    path = validate_file_path("data/input.csv", allowed_extensions=[".csv"])
    assert path.suffix == ".csv"


def test_validate_file_path_rejects_disallowed_extension() -> None:
    with pytest.raises(SecurityError, match="not allowed"):
        validate_file_path("data/input.exe", allowed_extensions=["csv", "parquet"])


def test_validate_file_path_rejects_traversal() -> None:
    with pytest.raises(SecurityError, match="Path traversal"):
        validate_file_path("../secrets.csv", allowed_extensions=[".csv"])


def test_hash_pii_columns_requires_salt() -> None:
    df = pd.DataFrame({"email": ["user@example.com"]})
    with pytest.raises(SecurityError, match="salt"):
        hash_pii_columns(df, ["email"])


def test_hash_pii_columns_hashes_values_and_preserves_na() -> None:
    df = pd.DataFrame({"email": ["user@example.com", None]})
    out = hash_pii_columns(df, ["email"], salt="test-salt")

    assert out.loc[0, "email"] != "user@example.com"
    assert len(out.loc[0, "email"]) == 16
    assert pd.isna(out.loc[1, "email"])
