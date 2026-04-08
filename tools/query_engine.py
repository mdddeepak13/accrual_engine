"""
Query engine — all DataFrame operations used by the conversational agent.
Each function returns a plain Python structure (list of dicts or a dict)
that is JSON-serialisable and easy for Claude to narrate.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ilike(series: pd.Series, value: str) -> pd.Series:
    """Case-insensitive substring match."""
    return series.str.contains(value, case=False, na=False)


def _apply_common_filters(df: pd.DataFrame, **kw) -> pd.DataFrame:
    if kw.get("period"):
        df = df[df["period"] == kw["period"]]
    if kw.get("company_code"):
        df = df[df["company_code"] == kw["company_code"]]
    if kw.get("company_name"):
        df = df[_ilike(df["company_name"], kw["company_name"])]
    if kw.get("gl_account"):
        df = df[df["gl_account"] == kw["gl_account"]]
    if kw.get("gl_description"):
        df = df[_ilike(df["gl_description"], kw["gl_description"])]
    if kw.get("vendor_name"):
        df = df[_ilike(df["vendor_name"], kw["vendor_name"])]
    if kw.get("vendor_number"):
        df = df[df["vendor_number"] == kw["vendor_number"]]
    if kw.get("min_amount") is not None:
        df = df[df["amount_usd"] >= kw["min_amount"]]
    if kw.get("max_amount") is not None:
        df = df[df["amount_usd"] <= kw["max_amount"]]
    return df


# ─── public query functions ────────────────────────────────────────────────────

def query_accruals(
    df: pd.DataFrame,
    period: str | None = None,
    company_code: str | None = None,
    company_name: str | None = None,
    gl_account: str | None = None,
    gl_description: str | None = None,
    vendor_name: str | None = None,
    vendor_number: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    sort_by: str = "amount_desc",  # amount_desc | amount_asc | period | company | gl
    limit: int = 25,
) -> dict:
    """
    Return filtered accrual rows plus aggregate totals.
    """
    filtered = _apply_common_filters(
        df.copy(),
        period=period,
        company_code=company_code,
        company_name=company_name,
        gl_account=gl_account,
        gl_description=gl_description,
        vendor_name=vendor_name,
        vendor_number=vendor_number,
        min_amount=min_amount,
        max_amount=max_amount,
    )

    sort_map = {
        "amount_desc": ("amount_usd", False),
        "amount_asc":  ("amount_usd", True),
        "period":      ("period",     True),
        "company":     ("company_name", True),
        "gl":          ("gl_description", True),
    }
    col, asc = sort_map.get(sort_by, ("amount_usd", False))
    filtered = filtered.sort_values(col, ascending=asc)

    total_rows = len(filtered)
    total_amount = float(filtered["amount_usd"].sum())
    avg_amount = float(filtered["amount_usd"].mean()) if total_rows else 0.0

    rows = filtered.head(limit)[
        ["sl_no", "company_name", "period", "gl_description", "vendor_name",
         "short_text", "amount_usd", "accrual_from", "accrual_to"]
    ].copy()
    rows["accrual_from"] = rows["accrual_from"].dt.strftime("%Y/%m/%d")
    rows["accrual_to"]   = rows["accrual_to"].dt.strftime("%Y/%m/%d")
    rows["amount_usd"]   = rows["amount_usd"].round(2)

    return {
        "total_matching_rows": total_rows,
        "total_amount_usd": round(total_amount, 2),
        "avg_amount_usd": round(avg_amount, 2),
        "rows_returned": min(limit, total_rows),
        "entries": rows.to_dict(orient="records"),
    }


def get_summary(
    df: pd.DataFrame,
    group_by: str,                            # gl_description | company_name | vendor_name | period
    period: str | None = None,
    company_name: str | None = None,
    gl_description: str | None = None,
    vendor_name: str | None = None,
    sort_by: str = "total_desc",              # total_desc | total_asc | count_desc | avg_desc
    limit: int = 20,
) -> dict:
    """
    Aggregate accruals by a chosen dimension and return ranked rows.
    """
    filtered = _apply_common_filters(
        df.copy(),
        period=period,
        company_name=company_name,
        gl_description=gl_description,
        vendor_name=vendor_name,
    )

    valid_groups = {"gl_description", "company_name", "vendor_name", "period",
                    "company_code", "gl_account"}
    if group_by not in valid_groups:
        return {"error": f"group_by must be one of: {sorted(valid_groups)}"}

    agg = (
        filtered.groupby(group_by)["amount_usd"]
        .agg(total="sum", count="count", avg="mean", min="min", max="max")
        .reset_index()
    )

    sort_col_map = {
        "total_desc": ("total", False),
        "total_asc":  ("total", True),
        "count_desc": ("count", False),
        "avg_desc":   ("avg",   False),
    }
    col, asc = sort_col_map.get(sort_by, ("total", False))
    agg = agg.sort_values(col, ascending=asc).head(limit)
    agg[["total", "avg", "min", "max"]] = agg[["total", "avg", "min", "max"]].round(2)

    return {
        "group_by": group_by,
        "period_filter": period,
        "total_entries": int(filtered.shape[0]),
        "grand_total_usd": round(float(filtered["amount_usd"].sum()), 2),
        "rows": agg.to_dict(orient="records"),
    }


def get_anomalies(
    anomalies: list[dict],
    period: str | None = None,
    severity: str | None = None,
    anomaly_type: str | None = None,
    company_name: str | None = None,
    gl_description: str | None = None,
    vendor_name: str | None = None,
    limit: int = 25,
) -> dict:
    """
    Filter and return anomaly records with a summary count.
    """
    result = anomalies

    if period:
        result = [a for a in result if a["period"] == period]
    if severity:
        result = [a for a in result if a["severity"].upper() == severity.upper()]
    if anomaly_type:
        result = [a for a in result if anomaly_type.upper() in a["anomaly_type"].upper()]
    if company_name:
        result = [a for a in result if company_name.lower() in a["company_name"].lower()]
    if gl_description:
        result = [a for a in result if gl_description.lower() in a["gl_description"].lower()]
    if vendor_name:
        result = [a for a in result if vendor_name.lower() in a["vendor_name"].lower()]

    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for a in result:
        sev = a["severity"]
        if sev in severity_counts:
            severity_counts[sev] += 1

    type_counts: dict[str, int] = {}
    for a in result:
        t = a["anomaly_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "total_anomalies": len(result),
        "severity_breakdown": severity_counts,
        "type_breakdown": type_counts,
        "entries": result[:limit],
    }


def get_suggested_accruals(
    suggestions: list[dict],
    target_period: str,
    company_name: str | None = None,
    gl_description: str | None = None,
    vendor_name: str | None = None,
    confidence: str | None = None,
    sort_by: str = "amount_desc",
    limit: int = 25,
) -> dict:
    """
    Filter and return suggested accruals for the target period.
    """
    result = suggestions

    if company_name:
        result = [s for s in result if company_name.lower() in s["company_name"].lower()]
    if gl_description:
        result = [s for s in result if gl_description.lower() in s["gl_description"].lower()]
    if vendor_name:
        result = [s for s in result if vendor_name.lower() in s["vendor_name"].lower()]
    if confidence:
        result = [s for s in result if s["confidence"].upper() == confidence.upper()]

    if sort_by == "amount_asc":
        result = sorted(result, key=lambda x: x["suggested_amount_usd"])
    else:
        result = sorted(result, key=lambda x: x["suggested_amount_usd"], reverse=True)

    total_amount = sum(s["suggested_amount_usd"] for s in result)
    confidence_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for s in result:
        c = s["confidence"]
        if c in confidence_counts:
            confidence_counts[c] += 1

    return {
        "target_period": target_period,
        "total_suggestions": len(result),
        "total_amount_usd": round(total_amount, 2),
        "confidence_breakdown": confidence_counts,
        "entries": result[:limit],
    }


def compare_periods(
    df: pd.DataFrame,
    period1: str,
    period2: str,
    group_by: str = "gl_description",
) -> dict:
    """
    Side-by-side comparison of two periods, showing amount change and % change.
    """
    def _agg(p: str) -> pd.DataFrame:
        sub = df[df["period"] == p]
        return (
            sub.groupby(group_by)["amount_usd"]
            .sum()
            .reset_index()
            .rename(columns={"amount_usd": "total"})
        )

    df1 = _agg(period1)
    df2 = _agg(period2)

    merged = pd.merge(df1, df2, on=group_by, how="outer", suffixes=(f"_{period1}", f"_{period2}"))
    merged = merged.fillna(0)

    col1 = f"total_{period1}"
    col2 = f"total_{period2}"

    merged["change_usd"] = (merged[col2] - merged[col1]).round(2)
    merged["change_pct"] = (
        ((merged[col2] - merged[col1]) / merged[col1].replace(0, np.nan)) * 100
    ).round(1).fillna(0)

    merged[col1] = merged[col1].round(2)
    merged[col2] = merged[col2].round(2)
    merged = merged.sort_values("change_usd", ascending=False)

    return {
        "period1": period1,
        "period2": period2,
        "group_by": group_by,
        "period1_total": round(float(df[df["period"] == period1]["amount_usd"].sum()), 2),
        "period2_total": round(float(df[df["period"] == period2]["amount_usd"].sum()), 2),
        "rows": merged.to_dict(orient="records"),
    }


def get_vendor_profile(
    df: pd.DataFrame,
    anomalies: list[dict],
    vendor_name: str,
) -> dict:
    """Return a full profile for a vendor: history, GL breakdown, anomalies."""
    vendor_df = df[_ilike(df["vendor_name"], vendor_name)]
    if vendor_df.empty:
        return {"error": f"No data found for vendor matching '{vendor_name}'"}

    actual_name = vendor_df["vendor_name"].iloc[0]
    vendor_number = vendor_df["vendor_number"].iloc[0]

    by_period = (
        vendor_df.groupby("period")["amount_usd"]
        .agg(total="sum", count="count")
        .reset_index()
        .sort_values("period")
        .round(2)
        .to_dict(orient="records")
    )
    by_gl = (
        vendor_df.groupby("gl_description")["amount_usd"]
        .sum()
        .reset_index()
        .sort_values("amount_usd", ascending=False)
        .rename(columns={"amount_usd": "total"})
        .round(2)
        .to_dict(orient="records")
    )
    by_company = (
        vendor_df.groupby("company_name")["amount_usd"]
        .sum()
        .reset_index()
        .sort_values("amount_usd", ascending=False)
        .rename(columns={"amount_usd": "total"})
        .round(2)
        .to_dict(orient="records")
    )
    vendor_anomalies = [a for a in anomalies if vendor_name.lower() in a["vendor_name"].lower()]

    return {
        "vendor_name": actual_name,
        "vendor_number": vendor_number,
        "total_amount_usd": round(float(vendor_df["amount_usd"].sum()), 2),
        "total_entries": int(len(vendor_df)),
        "periods_active": sorted(vendor_df["period"].unique().tolist()),
        "by_period": by_period,
        "by_gl_account": by_gl,
        "by_company": by_company,
        "anomalies": vendor_anomalies,
    }


def get_period_overview(df: pd.DataFrame, anomalies: list[dict], period: str) -> dict:
    """One-shot dashboard for a specific period."""
    period_df = df[df["period"] == period]
    if period_df.empty:
        available = sorted(df["period"].unique().tolist())
        return {"error": f"No data for period '{period}'. Available: {available}"}

    by_gl = (
        period_df.groupby("gl_description")["amount_usd"]
        .sum().sort_values(ascending=False).round(2).to_dict()
    )
    by_company = (
        period_df.groupby("company_name")["amount_usd"]
        .sum().sort_values(ascending=False).round(2).to_dict()
    )
    top_entries = (
        period_df.nlargest(5, "amount_usd")[
            ["company_name", "gl_description", "vendor_name", "amount_usd"]
        ].round(2).to_dict(orient="records")
    )
    period_anomalies = [a for a in anomalies if a.get("period") == period]

    return {
        "period": period,
        "total_entries": int(len(period_df)),
        "total_amount_usd": round(float(period_df["amount_usd"].sum()), 2),
        "avg_amount_usd": round(float(period_df["amount_usd"].mean()), 2),
        "by_gl_account": by_gl,
        "by_company": by_company,
        "top_5_entries": top_entries,
        "anomalies_count": len(period_anomalies),
        "anomalies": period_anomalies[:10],
    }
