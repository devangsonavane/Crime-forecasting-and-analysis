from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# --- Prepare LSTM Input: Group by Crime Type ---
type_monthly = df_clean.groupby(['Month_Year', 'Crm_Cd_Desc']).size().unstack(fill_value=0)
type_monthly.index = pd.to_datetime(type_monthly.index)
type_monthly = type_monthly.asfreq('MS')

# We'll forecast for one example crime type (e.g., 'VEHICLE - STOLEN') to demonstrate
example_crime = 'VEHICLE - STOLEN'
crime_series = type_monthly[example_crime].values.reshape(-1, 1)

# Normalize values
scaler = MinMaxScaler()
crime_scaled = scaler.fit_transform(crime_series)

# Create sequences
def create_sequences(data, seq_len=12):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(crime_scaled)

# Reshape input for LSTM [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=50, verbose=0)

# Forecast next 12 months
input_seq = crime_scaled[-12:].reshape(1, 12, 1)
preds = []
for _ in range(12):
    pred = model.predict(input_seq)[0]
    preds.append(pred)
    input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)

# Inverse scale
preds = scaler.inverse_transform(preds)

# Plot
forecast_index = pd.date_range(start=type_monthly.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
plt.figure(figsize=(12, 6))
plt.plot(type_monthly.index, crime_series, label='Observed')
plt.plot(forecast_index, preds, label='Forecast', linestyle='--', color='orange')
plt.title(f"LSTM Forecast for '{example_crime}' Crimes")
plt.xlabel("Date")
plt.ylabel("Number of Crimes")
plt.legend()
plt.tight_layout()
plt.show()
