"""Detect irregularities in accrual entries against historical baselines."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class Anomaly:
    sl_no: int
    company_code: str
    company_name: str
    gl_account: str
    gl_description: str
    vendor_number: str
    vendor_name: str
    amount_usd: float
    period: str
    severity: str          # HIGH / MEDIUM / LOW
    anomaly_type: str
    detail: str


def detect_anomalies(df: pd.DataFrame, baselines: dict, z_threshold: float = 2.0) -> list[dict]:
    """
    Flag accrual entries that deviate from historical baselines.

    Checks:
    1. Z-score outlier — amount deviates more than `z_threshold` std devs from the group mean.
    2. New vendor-GL combo — combination never seen in history.
    3. Zero / near-zero amount — likely a data entry error.
    4. Amount spike — more than 3x the historical max for the group.
    """
    anomalies: list[Anomaly] = []

    for _, row in df.iterrows():
        key = (row["company_code"], row["gl_account"], row["vendor_number"])
        amount = row["amount_usd"]
        period = row["period"]

        # Check 3 — zero or near-zero
        if amount <= 0:
            anomalies.append(Anomaly(
                sl_no=int(row["sl_no"]),
                company_code=row["company_code"],
                company_name=row["company_name"],
                gl_account=row["gl_account"],
                gl_description=row["gl_description"],
                vendor_number=row["vendor_number"],
                vendor_name=row["vendor_name"],
                amount_usd=amount,
                period=period,
                severity="HIGH",
                anomaly_type="ZERO_AMOUNT",
                detail=f"Amount is zero or negative ({amount:.2f}). Likely a data entry error.",
            ))
            continue

        if key not in baselines:
            # Check 2 — new vendor-GL combo
            anomalies.append(Anomaly(
                sl_no=int(row["sl_no"]),
                company_code=row["company_code"],
                company_name=row["company_name"],
                gl_account=row["gl_account"],
                gl_description=row["gl_description"],
                vendor_number=row["vendor_number"],
                vendor_name=row["vendor_name"],
                amount_usd=amount,
                period=period,
                severity="MEDIUM",
                anomaly_type="NEW_VENDOR_GL_COMBO",
                detail=(
                    f"Vendor {row['vendor_name']} ({row['vendor_number']}) has no prior history "
                    f"under GL {row['gl_description']} ({row['gl_account']}) for {row['company_name']}."
                ),
            ))
            continue

        b = baselines[key]
        mean, std, hist_max = b["mean"], b["std"], b["max"]

        # Check 4 — spike beyond 3× historical max
        if amount > 3 * hist_max:
            anomalies.append(Anomaly(
                sl_no=int(row["sl_no"]),
                company_code=row["company_code"],
                company_name=row["company_name"],
                gl_account=row["gl_account"],
                gl_description=row["gl_description"],
                vendor_number=row["vendor_number"],
                vendor_name=row["vendor_name"],
                amount_usd=amount,
                period=period,
                severity="HIGH",
                anomaly_type="AMOUNT_SPIKE",
                detail=(
                    f"Amount ${amount:,.2f} is more than 3× the historical max of ${hist_max:,.2f} "
                    f"for this vendor-GL combination."
                ),
            ))
            continue

        # Check 1 — Z-score
        if std > 0:
            z = abs((amount - mean) / std)
            if z >= z_threshold:
                severity = "HIGH" if z >= 3.0 else "MEDIUM"
                direction = "above" if amount > mean else "below"
                anomalies.append(Anomaly(
                    sl_no=int(row["sl_no"]),
                    company_code=row["company_code"],
                    company_name=row["company_name"],
                    gl_account=row["gl_account"],
                    gl_description=row["gl_description"],
                    vendor_number=row["vendor_number"],
                    vendor_name=row["vendor_name"],
                    amount_usd=amount,
                    period=period,
                    severity=severity,
                    anomaly_type="STATISTICAL_OUTLIER",
                    detail=(
                        f"Amount ${amount:,.2f} is {z:.1f}σ {direction} the historical mean of "
                        f"${mean:,.2f} (std=${std:,.2f}) for {row['vendor_name']} / "
                        f"{row['gl_description']} at {row['company_name']}."
                    ),
                ))

    return [asdict(a) for a in anomalies]
