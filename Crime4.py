import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import datetime

# ----------------------------
# LOAD & CLEAN DATA
# ----------------------------
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    df['DATE_OCC'] = pd.to_datetime(df['DATE_OCC'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df['Date_Rptd'] = pd.to_datetime(df['Date_Rptd'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df = df.dropna(subset=['DATE_OCC', 'Date_Rptd'])

    latest_month = df['DATE_OCC'].dt.to_period('M').max()
    cutoff_date = datetime(latest_month.year, latest_month.month, 1)
    df = df[df['DATE_OCC'] < cutoff_date]

    df['Crm_Cd_Desc'] = df['Crm_Cd_Desc'].str.upper().str.strip()
    df['Month_Year'] = df['DATE_OCC'].dt.to_period('M').astype(str)
    df['Month'] = df['DATE_OCC'].dt.month
    df['Weekday'] = df['DATE_OCC'].dt.weekday
    return df

# ----------------------------
# VISUALIZATIONS
# ----------------------------
def plot_visualizations(df):
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 6))

    # Monthly trend
    trend = df['Month_Year'].value_counts().sort_index()
    plt.plot(trend.index, trend.values, marker='o')
    plt.title("Monthly Crime Trends")
    plt.xticks(rotation=45)
    plt.xlabel("Month-Year")
    plt.ylabel("Number of Crimes")
    plt.tight_layout()
    plt.show()

# ----------------------------
# PROPHET FORECAST
# ----------------------------
def forecast_prophet(df):
    monthly_crime = df.groupby('Month_Year').size().reset_index(name='y')
    monthly_crime['ds'] = pd.to_datetime(monthly_crime['Month_Year'])

    prophet_model = Prophet()
    prophet_model.fit(monthly_crime[['ds', 'y']])

    future = prophet_model.make_future_dataframe(periods=12, freq='MS')
    forecast = prophet_model.predict(future)

    fig = prophet_model.plot(forecast)
    plt.title("Total Crime Forecast using Prophet")
    plt.xlabel("Date")
    plt.ylabel("Crimes")
    plt.tight_layout()
    plt.show()

# ----------------------------
# ARIMA FORECAST (Smooth & Connected)
# ----------------------------
def forecast_arima(df):
    ts = df.groupby('Month_Year').size()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.asfreq('MS')

    model = ARIMA(ts, order=(1,1,1))
    fitted = model.fit()

    forecast = fitted.get_forecast(steps=12)
    forecast_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    forecast_values = forecast.predicted_mean

    combined = pd.concat([ts, forecast_values])
    plt.figure(figsize=(14, 6))
    plt.plot(combined, label='Observed + Forecast (ARIMA)', color='blue')
    plt.axvline(x=forecast_index[0], color='gray', linestyle='--', label='Forecast Start')
    plt.title("ARIMA Forecast of Total Monthly Crimes")
    plt.xlabel("Date")
    plt.ylabel("Number of Crimes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------
# LSTM FORECAST (Multivariate)
# ----------------------------
def forecast_lstm(df, crime_type='VEHICLE - STOLEN'):
    df_type = df[df['Crm_Cd_Desc'] == crime_type]
    grouped = df_type.groupby(['Month_Year', 'Month', 'Weekday']).size().reset_index(name='CrimeCount')
    grouped['Month_Year'] = pd.to_datetime(grouped['Month_Year'])

    pivoted = grouped.groupby('Month_Year').agg({
        'CrimeCount': 'sum',
        'Month': 'first',
        'Weekday': 'mean'
    }).reset_index()

    pivoted['CrimeCount'] = pivoted['CrimeCount'].rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(pivoted[['CrimeCount', 'Month', 'Weekday']])

    def create_sequences(data, seq_len=36):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled)
    model = Sequential()
    model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(32, activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=1)

    input_seq = scaled[-36:].reshape(1, 36, 3)
    preds = []
    for _ in range(12):
        pred = model.predict(input_seq)[0][0]
        preds.append(pred)
        next_input = np.array([[pred, input_seq[0, -1, 1], input_seq[0, -1, 2]]])
        input_seq = np.append(input_seq[:, 1:, :], [next_input], axis=1)

    all_data = np.concatenate([scaled, np.array([[0, 0, 0]] * 12)], axis=0)
    all_data[-12:, 0] = preds
    reversed_preds = scaler.inverse_transform(all_data)[-12:, 0]

    forecast_dates = pd.date_range(start=pivoted['Month_Year'].iloc[0], periods=len(pivoted) + 12, freq='MS')
    combined_series = np.concatenate([pivoted['CrimeCount'].values, reversed_preds])

    plt.figure(figsize=(14, 6))
    plt.plot(forecast_dates, combined_series, label='Observed + Forecast (LSTM)', color='orange')
    plt.axvline(x=forecast_dates[len(pivoted)], color='gray', linestyle='--', label='Forecast Start')
    plt.title(f"LSTM Forecast for '{crime_type}' Crimes")
    plt.xlabel("Date")
    plt.ylabel("Crime Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# MAIN
# ----------------------------
if __name__ == '__main__':
    file_path = "E:/Programs/SEM 6/BDA/Project/Crime_Data_from_2020_to_Present.csv"
    df = load_and_prepare_data(file_path)

    plot_visualizations(df)
    forecast_prophet(df)
    forecast_arima(df)
    forecast_lstm(df, crime_type='VEHICLE - STOLEN')
