import pandas as pd
import numpy as np

# Load the uploaded CSV file
file_path = "E:\Programs\SEM 6\BDA\Project\Crime_Data_from_2020_to_Present.csv"
df = pd.read_csv(file_path)

# Display basic information and first few rows
df.info(), df.head()

# Create a copy to avoid modifying the original
df_clean = df.copy()

# Perform complete data cleaning and feature extraction for all columns

# Strip whitespace from column names
df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_')

# Re-convert date columns
df_clean['DATE_OCC'] = pd.to_datetime(df_clean['DATE_OCC'], errors='coerce')
df_clean['Date_Rptd'] = pd.to_datetime(df_clean['Date_Rptd'], errors='coerce')

# Drop rows with missing crucial date values
df_clean = df_clean.dropna(subset=['DATE_OCC', 'Date_Rptd'])

# Fill missing values for object (categorical) columns with 'Unknown'
cat_cols = df_clean.select_dtypes(include=['object']).columns
df_clean[cat_cols] = df_clean[cat_cols].fillna('Unknown')

# Fill missing values for numeric columns with -1
num_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[num_cols] = df_clean[num_cols].fillna(-1)

# Cast certain columns to category where appropriate
category_candidates = ['Vict_Sex', 'Vict_Descent', 'Status', 'Status_Desc', 'Part_1-2', 'AREA_NAME']
for col in category_candidates:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype('category')

# Time-based feature extraction
df_clean['Year'] = df_clean['DATE_OCC'].dt.year
df_clean['Month'] = df_clean['DATE_OCC'].dt.month
df_clean['Day'] = df_clean['DATE_OCC'].dt.day
df_clean['Hour'] = df_clean['TIME_OCC'].astype(int) // 100
df_clean['Weekday'] = df_clean['DATE_OCC'].dt.dayofweek

# Create a 'Month_Year' column for time series aggregation
df_clean['Month_Year'] = df_clean['DATE_OCC'].dt.to_period('M').astype(str)

# Check for unique values in crime code description
df_clean['Crm_Cd_Desc'] = df_clean['Crm_Cd_Desc'].str.upper().str.strip()

# Display cleaned and feature-enriched data
tools.display_dataframe_to_user(name="Fully Cleaned & Featured Crime Data", dataframe=df_clean)

import matplotlib.pyplot as plt
import seaborn as sns

# Set plot styles
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (12, 6)

# 1. Monthly Crime Trends
monthly_trend = df_clean['Month_Year'].value_counts().sort_index()

# 2. Crime by Location Heatmap
lat_lon = df_clean[(df_clean['LAT'] != -1) & (df_clean['LON'] != -1)]

# 3. Crime by Hour and Weekday
hourly_crime = df_clean['Hour'].value_counts().sort_index()
weekday_crime = df_clean['Weekday'].value_counts().sort_index()

# 4. Most Common Crime Types
top_crimes = df_clean['Crm_Cd_Desc'].value_counts().head(15)

# 5. Victim Demographics: Age Distribution
vict_age_dist = df_clean[df_clean['Vict_Age'] > 0]['Vict_Age']

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(18, 16))

# 1. Monthly Crime Trends
axes[0, 0].plot(monthly_trend.index, monthly_trend.values, marker='o')
axes[0, 0].set_title("Monthly Crime Trends")
axes[0, 0].set_xlabel("Month-Year")
axes[0, 0].set_ylabel("Number of Crimes")
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Crime Location Density Plot
sns.kdeplot(
    x=lat_lon['LON'], y=lat_lon['LAT'], cmap="Reds", fill=True, ax=axes[0, 1], thresh=0.05
)
axes[0, 1].set_title("Crime Density by Location")
axes[0, 1].set_xlabel("Longitude")
axes[0, 1].set_ylabel("Latitude")

# 3. Crime by Hour
sns.barplot(x=hourly_crime.index, y=hourly_crime.values, ax=axes[1, 0], palette='Blues_d')
axes[1, 0].set_title("Crimes by Hour of Day")
axes[1, 0].set_xlabel("Hour")
axes[1, 0].set_ylabel("Crime Count")

# 4. Crime by Weekday
weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
sns.barplot(x=weekday_labels, y=weekday_crime.values, ax=axes[1, 1], palette='Greens_d')
axes[1, 1].set_title("Crimes by Weekday")
axes[1, 1].set_xlabel("Weekday")
axes[1, 1].set_ylabel("Crime Count")

# 5. Top 15 Crime Types
sns.barplot(y=top_crimes.index, x=top_crimes.values, ax=axes[2, 0], palette='Reds_d')
axes[2, 0].set_title("Top 15 Most Common Crime Types")
axes[2, 0].set_xlabel("Crime Count")
axes[2, 0].set_ylabel("Crime Type")

# 6. Victim Age Distribution
sns.histplot(vict_age_dist, bins=30, kde=True, ax=axes[2, 1], color='purple')
axes[2, 1].set_title("Victim Age Distribution")
axes[2, 1].set_xlabel("Age")
axes[2, 1].set_ylabel("Count")

plt.tight_layout()
plt.show()

#total crime prediction
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import adfuller

register_matplotlib_converters()

# Prepare data for ARIMA: total monthly crime counts
monthly_crime = df_clean.groupby('Month_Year').size().sort_index()
monthly_crime.index = pd.to_datetime(monthly_crime.index)
monthly_crime = monthly_crime.asfreq('MS')  # ensure monthly start frequency

# Check stationarity using Augmented Dickey-Fuller test
adf_result = adfuller(monthly_crime)

# Differencing if needed
if adf_result[1] > 0.05:
    monthly_crime_diff = monthly_crime.diff().dropna()
else:
    monthly_crime_diff = monthly_crime

# Fit ARIMA model (auto parameters can be tuned later)
model = ARIMA(monthly_crime, order=(1,1,1))
model_fit = model.fit()

# Forecast for next 12 months
forecast = model_fit.forecast(steps=12)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(monthly_crime, label='Observed')
plt.plot(forecast.index, forecast.values, label='Forecast', color='blue', linestyle='--')
plt.title("ARIMA Forecast of Total Monthly Crimes")
plt.xlabel("Date")
plt.ylabel("Number of Crimes")
plt.legend()
plt.tight_layout()
plt.show()

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
