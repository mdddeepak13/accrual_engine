"""Analyze historical accrual patterns to derive baselines and trends."""

import json
import pandas as pd
import numpy as np


def compute_baselines(df: pd.DataFrame) -> dict:
    """
    Compute per-(company, GL account) statistics across all historical periods.
    Returns a dict keyed by (company_code, gl_account) with mean, std, min, max, count.
    """
    grouped = (
        df.groupby(["company_code", "company_name", "gl_account", "gl_description", "vendor_number", "vendor_name"])["amount_usd"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0)
    grouped["cv"] = grouped["std"] / grouped["mean"].replace(0, np.nan)  # coefficient of variation

    baselines = {}
    for _, row in grouped.iterrows():
        key = (row["company_code"], row["gl_account"], row["vendor_number"])
        baselines[key] = {
            "company_code": row["company_code"],
            "company_name": row["company_name"],
            "gl_account": row["gl_account"],
            "gl_description": row["gl_description"],
            "vendor_number": row["vendor_number"],
            "vendor_name": row["vendor_name"],
            "mean": round(row["mean"], 2),
            "std": round(row["std"], 2),
            "min": round(row["min"], 2),
            "max": round(row["max"], 2),
            "count": int(row["count"]),
            "cv": round(row["cv"], 4) if pd.notna(row["cv"]) else 0.0,
        }
    return baselines


def period_trends(df: pd.DataFrame) -> list[dict]:
    """Compute total accrual amount per period."""
    trend = (
        df.groupby("period")["amount_usd"]
        .agg(total="sum", count="count", average="mean")
        .reset_index()
        .sort_values("period")
    )
    return trend.to_dict(orient="records")


def top_accruals_by_gl(df: pd.DataFrame, period: str | None = None, top_n: int = 10) -> list[dict]:
    """Return top N GL accounts by total accrual amount."""
    src = df if period is None else df[df["period"] == period]
    result = (
        src.groupby(["gl_account", "gl_description"])["amount_usd"]
        .sum()
        .reset_index()
        .sort_values("amount_usd", ascending=False)
        .head(top_n)
    )
    return result.to_dict(orient="records")


def summary_by_company(df: pd.DataFrame, period: str | None = None) -> list[dict]:
    """Return accrual totals by company."""
    src = df if period is None else df[df["period"] == period]
    result = (
        src.groupby(["company_code", "company_name"])["amount_usd"]
        .agg(total="sum", count="count", average="mean")
        .reset_index()
        .sort_values("total", ascending=False)
    )
    return result.to_dict(orient="records")


def format_for_llm(baselines: dict, trends: list, top_gl: list, company_summary: list) -> str:
    """Serialize analysis results to a compact JSON string for the LLM."""
    return json.dumps(
        {
            "period_trends": trends,
            "top_gl_accounts": top_gl,
            "company_summary": company_summary,
            "baseline_count": len(baselines),
        },
        indent=2,
        default=str,
    )
