"""Load and normalize accrual data from Excel/CSV sources.

Accepts file paths (str/Path) or file-like objects (Streamlit UploadedFile / BytesIO).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import pandas as pd

# Columns expected in every accrual file
REQUIRED_COLUMNS = [
    "sl_no", "company_code", "company_name", "posting_date", "document_date",
    "gl_account", "gl_description", "vendor_number", "vendor_name",
    "short_text", "long_text", "accrual_from", "accrual_to", "amount_usd",
]

# Accepted input column names → normalised name
COLUMN_ALIASES: dict[str, str] = {
    # canonical
    "sl.no.": "sl_no", "sl no": "sl_no", "sl_no": "sl_no", "#": "sl_no",
    "company code": "company_code", "company_code": "company_code",
    "company name": "company_name", "company_name": "company_name",
    "posting date": "posting_date", "posting_date": "posting_date",
    "document date": "document_date", "document_date": "document_date",
    "gl account number": "gl_account", "gl account": "gl_account", "gl_account": "gl_account",
    "gl description": "gl_description", "gl_description": "gl_description",
    "vendor number": "vendor_number", "vendor_number": "vendor_number",
    "vendor name": "vendor_name", "vendor_name": "vendor_name",
    "short text": "short_text", "short_text": "short_text",
    "long text": "long_text", "long_text": "long_text",
    "accrual from period": "accrual_from", "accrual from": "accrual_from", "accrual_from": "accrual_from",
    "accrual to period": "accrual_to", "accrual to": "accrual_to", "accrual_to": "accrual_to",
    "amount (usd)": "amount_usd", "amount usd": "amount_usd",
    "amount": "amount_usd", "amount_usd": "amount_usd",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using COLUMN_ALIASES."""
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=rename_map)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, normalise period column, coerce amounts."""
    for col in ["posting_date", "document_date", "accrual_from", "accrual_to"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "accrual_from" in df.columns:
        df["period"] = df["accrual_from"].dt.strftime("%Y/%m")
    else:
        df["period"] = "Unknown"

    if "amount_usd" in df.columns:
        df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
        df = df.dropna(subset=["amount_usd"])

    # Add sl_no if missing
    if "sl_no" not in df.columns:
        df.insert(0, "sl_no", range(1, len(df) + 1))

    return df


def _read_single(
    source: Union[str, Path, "io.IOBase"],
    file_name: str = "",
) -> pd.DataFrame:
    """Read one Excel or CSV file and return a clean DataFrame."""
    name = (file_name or str(source)).lower()

    if name.endswith(".csv"):
        df = pd.read_csv(source)
    else:
        # Try "Accrual Data" sheet first, fall back to first sheet
        try:
            df = pd.read_excel(source, sheet_name="Accrual Data")
        except Exception:
            df = pd.read_excel(source, sheet_name=0)

    df = _normalise_columns(df)
    df = _clean(df)

    # Tag source file so users can trace rows back
    df["_source_file"] = file_name or str(source)
    return df


# ─── Public API ───────────────────────────────────────────────────────────────

def load_accrual_data(
    file_path: Union[str, Path],
    sheet_name: str = "Accrual Data",
) -> pd.DataFrame:
    """Load a single local Excel file (backward-compatible entry point)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return _read_single(path, file_name=path.name)


def load_from_uploads(uploaded_files: list) -> pd.DataFrame:
    """
    Load one or more Streamlit UploadedFile objects and concatenate them.
    Each file can be Excel (.xlsx / .xls) or CSV.
    Returns a single combined DataFrame with a '_source_file' column.
    """
    if not uploaded_files:
        raise ValueError("No files provided.")

    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for uf in uploaded_files:
        try:
            raw = io.BytesIO(uf.read())
            df = _read_single(raw, file_name=uf.name)
            frames.append(df)
        except Exception as exc:
            errors.append(f"{uf.name}: {exc}")

    if errors:
        raise ValueError("Could not read some files:\n" + "\n".join(errors))

    combined = pd.concat(frames, ignore_index=True)

    # Re-number sl_no sequentially across all files
    combined["sl_no"] = range(1, len(combined) + 1)
    return combined


def merge_datasets(existing: pd.DataFrame, new_upload: pd.DataFrame) -> pd.DataFrame:
    """Append new_upload rows to existing DataFrame, deduplicate, and re-index."""
    combined = pd.concat([existing, new_upload], ignore_index=True)

    # Deduplicate on company+GL+vendor+period+amount
    dedup_cols = [c for c in ["company_code", "gl_account", "vendor_number", "period", "amount_usd"]
                  if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")

    combined = combined.sort_values("period", na_position="last").reset_index(drop=True)
    combined["sl_no"] = range(1, len(combined) + 1)
    return combined


def get_periods(df: pd.DataFrame) -> list[str]:
    return sorted(df["period"].dropna().unique().tolist())


def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    return df[df["period"] == period].copy()
