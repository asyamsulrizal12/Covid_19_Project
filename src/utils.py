import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def save_report(fig, filename, subfolder="figures"):
    folder = os.path.join("reports", subfolder)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"Report saved: {path}")

def export_dataframe(df, filename, subfolder="tables"):
    folder = os.path.join("reports", subfolder)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    print(f"Data exported: {path}")

def export_narrative(text, filename="covid_forecasting_insights.md", subfolder="narratives"):
    folder = os.path.join("reports", subfolder)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Narrative saved: {path}")

def export_executive_summary(filename="executive_summary.md", subfolder="narratives"):
    summary = """# Executive Summary

## Context
Indonesia experienced two major COVID-19 waves: Delta (2021) and Omicron (2022).
Daily cases declined after 2022, but forecasting remains important for healthcare planning.

## Models Compared
- ARIMA: Best short-term accuracy.
- Prophet: Captures long-term seasonality, weaker short-term fit.
- LSTM: Flexible, but requires more tuning and larger datasets.

## Key Insights
- ARIMA is the most reliable for short-term forecasts.
- Prophet highlights broader pandemic waves.
- LSTM shows potential for future improvements.

## Conclusion
Forecasting COVID-19 cases is challenging due to irregular reporting.
Combining models provides complementary strengths for healthcare decision-making.
"""
    folder = os.path.join("reports", subfolder)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Executive summary saved: {path}")

def log_message(message, logfile="reports/log.txt"):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        f.write(f"{timestamp} {message}\n")
    print(f"Log updated: {logfile}")