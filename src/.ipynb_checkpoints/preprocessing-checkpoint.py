import pandas as pd
import numpy as np

def load_data(path):
    """Load CSV data and ensure datetime format."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

def add_lag_features(df, cols=["daily_cases","daily_deaths","daily_recovered"], lags=[1,7,14]):
    """Add lag features for selected columns."""
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def add_rolling_features(df, cols=["daily_cases"], windows=[7,14]):
    """Add rolling mean features."""
    for col in cols:
        for w in windows:
            df[f"{col}_roll{w}"] = df[col].rolling(w).mean()
    return df

def add_growth_rate(df):
    """Add daily growth rate feature."""
    df["case_growth_rate"] = df["daily_cases"].pct_change().replace([np.inf, -np.inf], np.nan)
    return df

def log_transform(df, cols=["daily_cases","daily_deaths","daily_recovered"]):
    """Apply log transform safely to reduce skewness."""
    for col in cols:
        safe_values = df[col].clip(lower=0).fillna(0)
        df[f"log_{col}"] = np.log1p(safe_values)
    return df

def preprocess(path):
    """Full preprocessing pipeline."""
    df = load_data(path)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_growth_rate(df)
    df = log_transform(df)
    return df
