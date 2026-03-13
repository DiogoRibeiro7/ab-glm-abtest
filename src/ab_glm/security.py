"""
Security and data validation utilities for ab-glm-abtest.

This module provides tools for:
- Input sanitization
- Data validation
- SQL injection prevention (if connecting to databases)
- Safe file handling
- PII detection and handling
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


class SecurityError(Exception):
    """Raised when security checks fail."""
    pass


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize column names to prevent injection attacks.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potentially unsafe column names

    Returns
    -------
    pd.DataFrame
        DataFrame with sanitized column names
    """
    # Allow only alphanumeric characters and underscores
    safe_pattern = re.compile(r'^[a-zA-Z0-9_]+$')

    new_columns = {}
    for col in df.columns:
        if not safe_pattern.match(str(col)):
            # Replace unsafe characters with underscore
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
            new_columns[col] = safe_name

    if new_columns:
        df = df.rename(columns=new_columns)

    return df


def validate_file_path(file_path: str, allowed_extensions: Optional[List[str]] = None) -> Path:
    """
    Validate file path for security.

    Parameters
    ----------
    file_path : str
        Path to validate
    allowed_extensions : List[str], optional
        List of allowed file extensions

    Returns
    -------
    Path
        Validated path object

    Raises
    ------
    SecurityError
        If path is unsafe
    """
    if not file_path:
        raise SecurityError("File path cannot be empty.")
    if "\x00" in file_path:
        raise SecurityError("Null byte detected in file path.")

    raw_path = Path(file_path)

    # Check for path traversal attempts before resolution.
    if any(part == ".." for part in raw_path.parts):
        raise SecurityError(f"Path traversal detected in {file_path}")

    path = raw_path.resolve(strict=False)

    # Check file extension
    if allowed_extensions:
        normalized_exts = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in allowed_extensions]
        if path.suffix.lower() not in normalized_exts:
            raise SecurityError(f"File extension {path.suffix} not allowed. Allowed: {allowed_extensions}")

    # Check if path is within working directory (optional, can be strict)
    # cwd = Path.cwd()
    # if not path.is_relative_to(cwd):
    #     raise SecurityError(f"Path {path} is outside working directory")

    return path


def detect_pii(df: pd.DataFrame, sample_size: int = 100) -> Dict[str, List[str]]:
    """
    Detect potential PII (Personally Identifiable Information) in data.

    Parameters
    ----------
    df : pd.DataFrame
        Data to check
    sample_size : int
        Number of rows to sample for checking

    Returns
    -------
    Dict[str, List[str]]
        Dictionary of potential PII columns by type
    """
    pii_patterns = {
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'phone': re.compile(r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$'),
        'ssn': re.compile(r'^\d{3}-\d{2}-\d{4}$'),
        'credit_card': re.compile(r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$'),
        'ip_address': re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
    }

    potential_pii = {pii_type: [] for pii_type in pii_patterns}

    # Sample data for checking
    sample_df = df.sample(min(sample_size, len(df))) if len(df) > sample_size else df

    for col in sample_df.columns:
        if sample_df[col].dtype == 'object':
            # Check string columns for PII patterns
            for pii_type, pattern in pii_patterns.items():
                if sample_df[col].astype(str).str.match(pattern).any():
                    potential_pii[pii_type].append(col)

    # Check for columns that might contain names
    name_keywords = ['name', 'first', 'last', 'surname', 'fname', 'lname']
    for col in df.columns:
        if any(keyword in str(col).lower() for keyword in name_keywords):
            if 'name' not in potential_pii:
                potential_pii['name'] = []
            potential_pii['name'].append(col)

    # Check for high cardinality columns that might be IDs
    for col in df.columns:
        if df[col].nunique() / len(df) > 0.95:  # More than 95% unique values
            if 'unique_id' not in potential_pii:
                potential_pii['unique_id'] = []
            potential_pii['unique_id'].append(col)

    # Filter out empty categories
    potential_pii = {k: v for k, v in potential_pii.items() if v}

    return potential_pii


def hash_pii_columns(df: pd.DataFrame, columns: List[str], salt: Optional[str] = None) -> pd.DataFrame:
    """
    Hash PII columns for privacy protection.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing PII
    columns : List[str]
        Columns to hash
    salt : str, optional
        Salt for hashing (should be stored securely)

    Returns
    -------
    pd.DataFrame
        DataFrame with hashed PII columns
    """
    df_copy = df.copy()

    if not salt:
        raise SecurityError("A non-empty salt is required for hashing PII columns.")

    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: hashlib.sha256(f"{x}{salt}".encode()).hexdigest()[:16]
                if pd.notna(x) else x
            )

    return df_copy


def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, type]) -> List[str]:
    """
    Validate that columns have expected data types.

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
    expected_types : Dict[str, type]
        Expected types for each column

    Returns
    -------
    List[str]
        List of validation errors
    """
    errors = []

    for col, expected_type in expected_types.items():
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
            continue

        actual_dtype = df[col].dtype

        if expected_type == int:
            if not pd.api.types.is_integer_dtype(actual_dtype):
                errors.append(f"Column {col} should be integer, got {actual_dtype}")
        elif expected_type == float:
            if not pd.api.types.is_numeric_dtype(actual_dtype):
                errors.append(f"Column {col} should be numeric, got {actual_dtype}")
        elif expected_type == bool:
            # Check if values are 0/1 or True/False
            unique_vals = df[col].dropna().unique()
            if not set(unique_vals).issubset({0, 1, True, False}):
                errors.append(f"Column {col} should be boolean, got values: {unique_vals[:5]}")
        elif expected_type == str:
            if not pd.api.types.is_string_dtype(actual_dtype) and not pd.api.types.is_object_dtype(actual_dtype):
                errors.append(f"Column {col} should be string, got {actual_dtype}")

    return errors


def check_data_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data integrity checks.

    Parameters
    ----------
    df : pd.DataFrame
        Data to check

    Returns
    -------
    Dict[str, Any]
        Integrity check results
    """
    integrity = {
        'rows': len(df),
        'columns': len(df.columns),
        'duplicates': df.duplicated().sum(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'warnings': []
    }

    # Check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        integrity['warnings'].append(f"Duplicate column names: {duplicate_cols}")

    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        integrity['warnings'].append(f"Constant columns (no variation): {constant_cols}")

    # Check for extreme values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].nunique() > 1:  # Skip constant columns
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                integrity['warnings'].append(f"Column {col} has {outliers} extreme outliers")

    # Check for suspicious patterns
    for col in df.columns:
        # Check for SQL injection patterns
        if df[col].dtype == 'object':
            sql_patterns = ['DROP TABLE', 'DELETE FROM', 'INSERT INTO', '<script>', '<?php']
            for pattern in sql_patterns:
                if df[col].astype(str).str.contains(pattern, case=False, na=False).any():
                    integrity['warnings'].append(f"Potential SQL injection in column {col}")
                    break

    return integrity


def create_safe_summary(df: pd.DataFrame, include_samples: bool = False) -> Dict[str, Any]:
    """
    Create a safe summary of the data without exposing sensitive information.

    Parameters
    ----------
    df : pd.DataFrame
        Data to summarize
    include_samples : bool
        Whether to include sample values (be careful with PII)

    Returns
    -------
    Dict[str, Any]
        Safe summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'numeric_summary': {}
    }

    # Add numeric summaries
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median())
        }

    # Add categorical summaries (without actual values)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    summary['categorical_summary'] = {}
    for col in categorical_cols:
        summary['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'most_common_count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
        }

    if include_samples and len(df) > 0:
        # Only include hashed versions of sample data
        sample_size = min(5, len(df))
        summary['sample_hashes'] = {}
        for col in df.columns[:5]:  # Limit to first 5 columns
            sample_values = df[col].dropna().iloc[:sample_size]
            summary['sample_hashes'][col] = [
                hashlib.md5(str(val).encode()).hexdigest()[:8]
                for val in sample_values
            ]

    return summary


def validate_ab_test_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive validation for A/B test data.

    Parameters
    ----------
    df : pd.DataFrame
        A/B test data

    Returns
    -------
    Dict[str, Any]
        Validation results and recommendations
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': []
    }

    # Check required columns
    required_columns = {'user_id', 'T', 'y'}
    missing = required_columns - set(df.columns)
    if missing:
        results['valid'] = False
        results['errors'].append(f"Missing required columns: {missing}")

    # Sanitize column names
    original_columns = df.columns.tolist()
    df_safe = sanitize_column_names(df)
    if df_safe.columns.tolist() != original_columns:
        results['warnings'].append("Column names were sanitized for security")

    # Check for PII
    pii_detected = detect_pii(df)
    if pii_detected:
        results['warnings'].append(f"Potential PII detected: {pii_detected}")
        results['recommendations'].append("Consider hashing or removing PII columns")

    # Check data types
    if 'T' in df.columns:
        t_values = df['T'].unique()
        if not set(t_values).issubset({0, 1}):
            results['errors'].append(f"Treatment column has invalid values: {t_values}")
            results['valid'] = False

    if 'y' in df.columns:
        y_values = df['y'].unique()
        if not set(y_values).issubset({0, 1}):
            results['errors'].append(f"Outcome column has invalid values: {y_values}")
            results['valid'] = False

    # Check data integrity
    integrity = check_data_integrity(df)
    if integrity['warnings']:
        results['warnings'].extend(integrity['warnings'])

    # Check for treatment consistency
    if 'user_id' in df.columns and 'T' in df.columns:
        inconsistent = df.groupby('user_id')['T'].nunique()
        if (inconsistent > 1).any():
            n_inconsistent = (inconsistent > 1).sum()
            results['errors'].append(f"{n_inconsistent} users have inconsistent treatment assignment")
            results['valid'] = False

    # Check sample size
    if 'user_id' in df.columns:
        n_users = df['user_id'].nunique()
        if n_users < 100:
            results['warnings'].append(f"Small sample size: only {n_users} users")
            results['recommendations'].append("Consider collecting more data for reliable results")

    return results


if __name__ == "__main__":
    # Example usage and testing
    import pandas as pd

    # Create sample data with potential issues
    sample_data = pd.DataFrame({
        'user_id': range(100),
        'email': ['user@example.com'] * 100,  # PII
        'T': [0, 1] * 50,
        'y': [0, 1, 0, 1] * 25,
        'country_EU': [1, 0] * 50,
        'device_mobile': [1, 0, 1] * 33 + [1],
        'prior_views': range(100)
    })

    # Run validation
    validation_results = validate_ab_test_data(sample_data)
    print("Validation Results:")
    print(f"  Valid: {validation_results['valid']}")
    print(f"  Errors: {validation_results['errors']}")
    print(f"  Warnings: {validation_results['warnings']}")
    print(f"  Recommendations: {validation_results['recommendations']}")
