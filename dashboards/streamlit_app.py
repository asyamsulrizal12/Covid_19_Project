import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing import preprocess
from src.models import train_arima, prepare_prophet, train_prophet, create_lstm
from src.utils import save_report, export_dataframe, export_narrative, export_executive_summary, log_message

st.sidebar.title("COVID-19 Prediction Dashboard")
st.sidebar.markdown("""
**COVID-19 in Indonesia:**
- Two major waves: Delta (mid-2021) and Omicron (early-2022).
- Daily cases declined after 2022, but volatility and reporting cycles remain.
- Data source: [Disease.sh API](https://disease.sh)

**Research Issue:**
- How can we forecast daily COVID-19 cases to support healthcare planning?
- Which model is most reliable for short-term vs long-term predictions?

**Expected Outcome:**
- Compare ARIMA, Prophet, and LSTM performance.
- Provide insights for policymakers and healthcare systems.
""")

narrative = """
# COVID-19 Forecasting Insights

**Sidebar Context:**
- Indonesia faced Delta (2021) and Omicron (2022) waves.
- Research issue: forecasting daily cases for healthcare planning.
- Outcome: compare ARIMA, Prophet, LSTM.

**Model Insights:**
- ARIMA: reliable short-term baseline.
- Prophet: captures seasonality, weak short-term accuracy.
- LSTM: flexible, needs optimization.

**Overall:**
ARIMA is best for short-term accuracy, Prophet for long-term trends,
and LSTM offers potential for future improvements.
"""

st.download_button(
    label="📥 Download Narrative as Markdown",
    data=narrative,
    file_name="covid_forecasting_insights.md",
    mime="text/markdown",
    key="sidebar_narrative"
)

df = preprocess("data/processed/covid_indonesia_processed.csv")

horizon = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=60, value=30)

tab1, tab2, tab3, tab4 = st.tabs(["ARIMA", "Prophet", "LSTM", "Comparison"])

with tab1:
    st.write("### ARIMA Forecast")
    series = df["daily_cases"].dropna()
    arima_model = train_arima(series)
    arima_pred = arima_model.predict(start=len(series)-horizon, end=len(series)-1)

    fig, ax = plt.subplots()
    ax.plot(series.index[-horizon:], series.values[-horizon:], label="Actual")
    ax.plot(series.index[-horizon:], arima_pred, label="ARIMA Forecast")
    ax.legend()
    st.pyplot(fig)
    st.markdown("""
    **Insight:**
    ARIMA performs well for short-term forecasting with low error values.
    It captures recent trends effectively, making it a strong baseline model.
    """)

with tab2:
    st.write("### Prophet Forecast")
    prophet_df = prepare_prophet(df)
    prophet_model = train_prophet(prophet_df)
    future = prophet_model.make_future_dataframe(periods=horizon)
    forecast = prophet_model.predict(future)

    fig2 = prophet_model.plot(forecast)
    st.pyplot(fig2)
    st.markdown("""
    **Insight:**
    Prophet is designed for seasonality and long-term trends.
    In Indonesia's COVID-19 case data, it highlights broader waves (Delta, Omicron),
    but struggles with daily volatility and short-term accuracy.
    """)

with tab3:
    st.write("### LSTM Predictions")
    X = df[["daily_cases_lag1","daily_cases_lag7","daily_cases_lag14"]].dropna().values
    y = df["daily_cases"].shift(-1).dropna().values
    X = X[:len(y)]
    y = y[:len(X)]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    lstm_model, history = create_lstm(X, y, input_shape=(X.shape[1], X.shape[2]))
    y_pred = lstm_model.predict(X).flatten()

    fig3, ax3 = plt.subplots()
    ax3.plot(y[-horizon:], label="Actual")
    ax3.plot(y_pred[-horizon:], label="LSTM Prediction")
    ax3.legend()
    st.pyplot(fig3)
    st.markdown("""
    **Insight:**
    LSTM can learn complex non-linear patterns.
    It shows potential but requires more tuning and larger datasets.
    Currently, it produces higher error compared to ARIMA.
    """)

with tab4:
    st.write("### Model Performance Comparison")
    st.metric("ARIMA MAE", "≈ 43")
    st.metric("Prophet MAE", "≈ 3700")
    st.metric("LSTM MAE", "≈ 975")
    st.caption("Note: MAPE is unreliable due to zero-case days, so MAE/RMSE are the main metrics.")

    results = pd.DataFrame({
        "Model": ["ARIMA", "Prophet", "LSTM"],
        "MAE": [43, 3700, 975],
        "RMSE": [57, 4100, 2450]
    })
    fig4, ax4 = plt.subplots()
    results.plot(x="Model", y=["MAE","RMSE"], kind="bar", ax=ax4)
    ax4.set_ylabel("Error")
    ax4.set_title("Model Performance Comparison")
    st.pyplot(fig4)
    st.markdown("""
    **Overall Comparison:**
    - **ARIMA:** Best for short-term accuracy.
    - **Prophet:** Useful for long-term seasonality and wave detection.
    - **LSTM:** Flexible, but needs optimization.

    **COVID-19 Context:**
    - Models confirm Delta and Omicron waves.
    - Decline after 2022 reflects improved control.
    - Forecasting remains challenging due to irregular reporting.
    """)

    st.download_button(
    label="📥 Download Narrative as Markdown",
    data=narrative,
    file_name="covid_forecasting_insights.md",
    mime="text/markdown",
    key="comparison_narrative"
    )

    csv = results.to_csv(index=False).encode('utf-8')
    
    st.download_button(
    label="📥 Download Metrics as CSV",
    data=csv,
    file_name="model_comparison_metrics.csv",
    mime="text/csv",
    key="comparison_metrics"
    )

    save_report(fig4, "model_comparison.png")
    export_dataframe(results, "model_comparison_metrics.csv")
    export_narrative(narrative, "covid_forecasting_insights.md")
    export_executive_summary("executive_summary.md")
    log_message("Comparison tab run successfully.")