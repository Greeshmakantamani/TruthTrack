"""
metrics.py - compute basic metrics from a company CSV.

Expected columns (one row per quarter, oldest first):
  quarter, period_end, revenue_cr, volume_growth_pct,
  receivables_cr, inventory_cr, source_file, page

Usage:  python metrics.py data/britannia.csv
"""
import sys
import pandas as pd


def load(path):
    df = pd.read_csv(path, parse_dates=["period_end"])
    df = df.sort_values("period_end").reset_index(drop=True)
    for c in ["revenue_cr", "volume_growth_pct", "receivables_cr", "inventory_cr"]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_metrics(df):
    # days in each quarter (from the previous calendar quarter end)
    prev_end = df["period_end"] - pd.offsets.QuarterEnd(1)
    df["days_in_qtr"] = (df["period_end"] - prev_end).dt.days

    # year-on-year growth: compare with the row 4 quarters earlier
    df["revenue_yoy_pct"] = (df["revenue_cr"] / df["revenue_cr"].shift(4) - 1) * 100

    # working-capital days (only where the balance sheet exists)
    df["receivable_days"] = df["receivables_cr"] / df["revenue_cr"] * df["days_in_qtr"]
    df["inventory_days"] = df["inventory_cr"] / df["revenue_cr"] * df["days_in_qtr"]

    # price/realisation growth ~ revenue growth minus volume growth.
    # Only valid when volume growth is disclosed; otherwise left empty.
    df["price_realisation_pct"] = df["revenue_yoy_pct"] - df["volume_growth_pct"]
    return df


def flags(df):
    """Simple, rule-based observations. These are leads to investigate, not conclusions."""
    out = []
    g = df["revenue_yoy_pct"].dropna()
    if len(g) >= 3:
        med = g.median()
        for i, v in g.items():
            if abs(v - med) > 3:
                out.append(f"{df.loc[i,'quarter']}: revenue growth {v:.1f}% is far from median {med:.1f}%")
    r = df["receivable_days"].dropna()
    if len(r) >= 2 and r.iloc[-1] - r.iloc[0] > 5:
        out.append(f"Receivable days rose {r.iloc[0]:.1f} -> {r.iloc[-1]:.1f}")
    if df["volume_growth_pct"].isna().all():
        out.append("No volume data: price vs volume split not possible (add volume_growth_pct)")
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/britannia.csv"
    df = add_metrics(load(path))
    cols = ["quarter", "revenue_cr", "revenue_yoy_pct", "volume_growth_pct",
            "price_realisation_pct", "receivable_days", "inventory_days"]
    print(df[cols].round(2).to_string(index=False))
    print("\nFlags:")
    for f in flags(df) or ["none"]:
        print(" -", f)
