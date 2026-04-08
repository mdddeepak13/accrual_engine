"""Suggest accruals for the next period based on historical baselines and trends."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class SuggestedAccrual:
    company_code: str
    company_name: str
    gl_account: str
    gl_description: str
    vendor_number: str
    vendor_name: str
    suggested_amount_usd: float
    basis: str               # e.g. "6-period average", "last period + 3% trend"
    confidence: str          # HIGH / MEDIUM / LOW
    historical_mean: float
    historical_std: float
    periods_observed: int


def suggest_next_period(df: pd.DataFrame, baselines: dict, target_period: str) -> list[dict]:
    """
    For each historically observed (company, GL, vendor) combination, suggest
    an accrual amount for `target_period`.

    Strategy:
    - HIGH confidence  → 3+ periods of data, low coefficient of variation (CV < 0.25)
      → use the mean of the last 3 periods.
    - MEDIUM confidence → 2+ periods, or CV between 0.25–0.50
      → use overall mean with a note.
    - LOW confidence  → single observation, or CV > 0.50
      → use last known amount with a warning.
    """
    # Build per-combination time series
    ts = (
        df.sort_values("period")
        .groupby(["company_code", "company_name", "gl_account", "gl_description",
                  "vendor_number", "vendor_name", "period"])["amount_usd"]
        .mean()
        .reset_index()
    )

    suggestions: list[SuggestedAccrual] = []

    for key, b in baselines.items():
        company_code, gl_account, vendor_number = key
        mask = (
            (ts["company_code"] == company_code)
            & (ts["gl_account"] == gl_account)
            & (ts["vendor_number"] == vendor_number)
        )
        series = ts[mask].sort_values("period")["amount_usd"].tolist()
        n = len(series)
        mean = b["mean"]
        std = b["std"]
        cv = b["cv"]

        if n >= 3 and cv < 0.30:
            last3 = series[-3:]
            suggested = round(np.mean(last3), 2)
            basis = f"Mean of last {min(3, n)} periods (CV={cv:.2f}, low variability)"
            confidence = "HIGH"
        elif n >= 3 and cv < 0.60:
            suggested = round(mean, 2)
            basis = f"Overall {n}-period mean (CV={cv:.2f}, moderate variability)"
            confidence = "HIGH"
        elif n >= 2:
            suggested = round(mean, 2)
            basis = f"Overall {n}-period mean"
            confidence = "MEDIUM"
        else:
            suggested = round(series[-1], 2)
            basis = "Single observation (last known amount)"
            confidence = "LOW"

        suggestions.append(SuggestedAccrual(
            company_code=b["company_code"],
            company_name=b["company_name"],
            gl_account=b["gl_account"],
            gl_description=b["gl_description"],
            vendor_number=b["vendor_number"],
            vendor_name=b["vendor_name"],
            suggested_amount_usd=suggested,
            basis=basis,
            confidence=confidence,
            historical_mean=mean,
            historical_std=std,
            periods_observed=n,
        ))

    # Sort by company, then GL account, then amount descending
    suggestions.sort(key=lambda s: (s.company_code, s.gl_account, -s.suggested_amount_usd))
    return [asdict(s) for s in suggestions]
