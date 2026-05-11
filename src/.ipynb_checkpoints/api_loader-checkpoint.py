import requests
import pandas as pd
import os

API_URL = "https://disease.sh/v3/covid-19/historical/Indonesia?lastdays=all"

def fetch_covid_data():
    response = requests.get(API_URL)
    data = response.json()

    cases = data["timeline"]["cases"]
    deaths = data["timeline"]["deaths"]
    recovered = data["timeline"]["recovered"]

    df = pd.DataFrame({
        "date": pd.to_datetime(list(cases.keys()), format="%m/%d/%y"),
        "cases": list(cases.values()),
        "deaths": list(deaths.values()),
        "recovered": list(recovered.values())
    }).sort_values("date")

    return df

def save_raw_data(df, filename="covid_indonesia.csv"):
    raw_path = os.path.join("data", "raw")
    os.makedirs(raw_path, exist_ok=True)
    df.to_csv(os.path.join(raw_path, filename), index=False)
    print(f"Data saved to {raw_path}/{filename}")

if __name__ == "__main__":
    df = fetch_covid_data()
    save_raw_data(df)