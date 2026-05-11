import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_arima(series, order=(2,1,2)):
    """Train ARIMA model on a time series."""
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted

def prepare_prophet(df):
    """Prepare dataframe for Prophet (ds, y)."""
    prophet_df = df[["date","daily_cases"]].rename(columns={"date":"ds","daily_cases":"y"})
    return prophet_df

def train_prophet(prophet_df):
    """Train Prophet model."""
    model = Prophet()
    model.fit(prophet_df)
    return model

def create_lstm(X_train, y_train, input_shape, epochs=20, batch_size=32):
    """Build and train a simple LSTM model, return model + history."""
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history