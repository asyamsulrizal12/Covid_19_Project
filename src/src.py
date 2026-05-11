import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_correlation(df):
    """Plot correlation heatmap for main COVID-19 metrics."""
    corr = df[["cases","deaths","recovered",
               "daily_cases","daily_deaths","daily_recovered"]].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def plot_daily_cases(df):
    """Plot daily cases with 7-day and 14-day moving averages."""
    df["ma7"] = df["daily_cases"].rolling(7).mean()
    df["ma14"] = df["daily_cases"].rolling(14).mean()

    plt.figure(figsize=(14,6))
    sns.lineplot(x="date", y="daily_cases", data=df, label="Daily Cases")
    sns.lineplot(x="date", y="ma7", data=df, label="7-day MA")
    sns.lineplot(x="date", y="ma14", data=df, label="14-day MA")
    plt.legend()
    plt.title("COVID-19 Indonesia Daily Cases with Moving Average")
    plt.show()

def plot_distribution(df):
    """Plot histogram and boxplot for daily cases distribution."""
    plt.figure(figsize=(10,5))
    sns.histplot(df["daily_cases"], bins=50, kde=True)
    plt.title("Daily Cases Distribution")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.boxplot(y=df["daily_cases"])
    plt.title("Daily Cases Outlier")
    plt.show()

def plot_seasonality(df):
    """Decompose daily cases into trend, seasonality, and residuals."""
    result = seasonal_decompose(df["daily_cases"], model="additive", period=7)
    result.plot()
    plt.show()
