# COVID-19 Prediction Dashboard

## 📌 Project Overview
This project forecasts daily COVID-19 cases in Indonesia using **ARIMA**, **Prophet**, and **LSTM** models.  
It combines data collection, preprocessing, exploratory analysis, model training, and visualization into a complete workflow.

## 📂 Repository Structure
<img width="510" height="556" alt="image" src="https://github.com/user-attachments/assets/060817c1-6e58-4494-b0df-b3608ec18577" />

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/asyamsulrizal12/Covid_19_Project.git
   cd covid19-prediction
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Streamlit dashboard:
   streamlit run dashboards/streamlit_app.py

## 📊 Reports
a. Figures: model comparison, trend, seasonality, growth rate, etc.\
b. Tables: error metrics, forecast results.\
c. Narratives: technical insights (covid_forecasting_insights.md) and executive summary (executive_summary.md).\
d. Log: run history (log.txt).

## 🛠️ Models
a. ARIMA → best short-term accuracy.\
b. Prophet → captures long-term seasonality.\
c. LSTM → flexible, requires tuning.

## 📈 Data Source
Disease.sh API — global COVID-19 data
