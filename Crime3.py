import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from datetime import datetime

# --- LOAD DATA ---
df = pd.read_csv("E:/Programs/SEM 6/BDA/Project/Crime_Data_from_2020_to_Present.csv")

# --- CLEANING ---
df.columns = df.columns.str.strip().str.replace(" ", "_")
df['DATE_OCC'] = pd.to_datetime(df['DATE_OCC'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Date_Rptd'] = pd.to_datetime(df['Date_Rptd'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df = df.dropna(subset=['DATE_OCC', 'Date_Rptd'])

# Remove rows from incomplete latest month
latest_month = df['DATE_OCC'].dt.to_period('M').max()
cutoff_date = datetime(latest_month.year, latest_month.month, 1)
df = df[df['DATE_OCC'] < cutoff_date]

df['Crm_Cd_Desc'] = df['Crm_Cd_Desc'].str.upper().str.strip()
df['Month_Year'] = df['DATE_OCC'].dt.to_period('M').astype(str)
df['Month'] = df['DATE_OCC'].dt.month
df['Weekday'] = df['DATE_OCC'].dt.weekday

# --- PROPHET for TOTAL CRIMES ---
from statsmodels.tsa.arima.model import ARIMA

# --- ARIMA Forecast for Total Monthly Crimes ---
# Prepare the time series
ts = df.groupby('Month_Year').size()
ts.index = pd.to_datetime(ts.index)
ts = ts.asfreq('MS')  # Ensure monthly frequency

# Fit ARIMA
arima_model = ARIMA(ts, order=(1,1,1))
arima_result = arima_model.fit()

# Forecast next 12 months
forecast_steps = 12
forecast = arima_result.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
forecast_values = forecast.predicted_mean

# Combine observed + forecast
combined = pd.concat([ts, forecast_values])

# Plotting
plt.figure(figsize=(16, 8))
plt.plot(combined, label='Observed + Forecast (ARIMA)', color='blue')
plt.axvline(x=forecast_index[0], color='gray', linestyle='--', label='Forecast Start')
plt.title("ARIMA Forecast of Total Monthly Crimes")
plt.xlabel("Date")
plt.ylabel("Number of Crimes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- MULTIVARIATE LSTM for 'VEHICLE - STOLEN' ---
crime_type = 'VEHICLE - STOLEN'
df_type = df[df['Crm_Cd_Desc'] == crime_type]
grouped = df_type.groupby(['Month_Year', 'Month', 'Weekday']).size().reset_index(name='CrimeCount')
grouped['Month_Year'] = pd.to_datetime(grouped['Month_Year'])

# Pivot features
pivoted = grouped.groupby('Month_Year').agg({
    'CrimeCount': 'sum',
    'Month': 'first',
    'Weekday': 'mean'
}).reset_index()

# Smooth the data (optional)
pivoted['CrimeCount'] = pivoted['CrimeCount'].rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')

# Scale features
scaler = MinMaxScaler()
scaled = scaler.fit_transform(pivoted[['CrimeCount', 'Month', 'Weekday']])

# Create sequences
def create_sequences(data, seq_len=36):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)

# LSTM Model
model = Sequential()
model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(32, activation='tanh'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=1)

# Forecast
input_seq = scaled[-36:].reshape(1, 36, 3)
preds = []
for _ in range(12):
    pred = model.predict(input_seq)[0][0]
    preds.append(pred)
    next_input = np.array([[pred, input_seq[0, -1, 1], input_seq[0, -1, 2]]])
    input_seq = np.append(input_seq[:, 1:, :], [next_input], axis=1)

# Inverse transform predictions
all_data = np.concatenate([scaled, np.array([[0, 0, 0]] * 12)], axis=0)
all_data[-12:, 0] = preds
reversed_preds = scaler.inverse_transform(all_data)[-12:, 0]

# Plot full connected series
forecast_dates = pd.date_range(start=pivoted['Month_Year'].iloc[0], periods=len(pivoted) + 12, freq='MS')
combined_series = np.concatenate([pivoted['CrimeCount'].values, reversed_preds])

plt.figure(figsize=(12, 6))
plt.plot(forecast_dates, combined_series, label='Crime Count (Observed + Forecast)', color='orange')
plt.axvline(x=forecast_dates[len(pivoted)], color='gray', linestyle='--', label='Forecast Start')
plt.title(f"LSTM Forecast for '{crime_type}' Crimes (Multivariate + Connected)")
plt.xlabel("Date")
plt.ylabel("Crime Count")
plt.legend()
plt.tight_layout()
plt.show()