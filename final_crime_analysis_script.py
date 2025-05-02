
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import adfuller

# Load the uploaded CSV file
file_path = "E:/Programs/SEM 6/BDA/Project/Crime_Data_from_2020_to_Present.csv"
df = pd.read_csv(file_path)

# Create a copy to avoid modifying the original
df_clean = df.copy()

# Strip whitespace from column names
df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_')

# Re-convert date columns
df_clean['DATE_OCC'] = pd.to_datetime(df_clean['DATE_OCC'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df_clean['Date_Rptd'] = pd.to_datetime(df_clean['Date_Rptd'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')


# Drop rows with missing crucial date values
df_clean = df_clean.dropna(subset=['DATE_OCC', 'Date_Rptd'])

# Fill missing values
cat_cols = df_clean.select_dtypes(include=['object']).columns
df_clean[cat_cols] = df_clean[cat_cols].fillna('Unknown')
num_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[num_cols] = df_clean[num_cols].fillna(-1)

# Cast to category
category_candidates = ['Vict_Sex', 'Vict_Descent', 'Status', 'Status_Desc', 'Part_1-2', 'AREA_NAME']
for col in category_candidates:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype('category')

# Time-based features
df_clean['Year'] = df_clean['DATE_OCC'].dt.year
df_clean['Month'] = df_clean['DATE_OCC'].dt.month
df_clean['Day'] = df_clean['DATE_OCC'].dt.day
df_clean['Hour'] = df_clean['TIME_OCC'].astype(int) // 100
df_clean['Weekday'] = df_clean['DATE_OCC'].dt.dayofweek
df_clean['Month_Year'] = df_clean['DATE_OCC'].dt.to_period('M').astype(str)
df_clean['Crm_Cd_Desc'] = df_clean['Crm_Cd_Desc'].str.upper().str.strip()

# Set plot style
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (12, 6)

# Monthly trend
monthly_trend = df_clean['Month_Year'].value_counts().sort_index()
lat_lon = df_clean[(df_clean['LAT'] != -1) & (df_clean['LON'] != -1)]
hourly_crime = df_clean['Hour'].value_counts().sort_index()
weekday_crime = df_clean['Weekday'].value_counts().sort_index()
top_crimes = df_clean['Crm_Cd_Desc'].value_counts().head(15)
vict_age_dist = df_clean[df_clean['Vict_Age'] > 0]['Vict_Age']

fig, axes = plt.subplots(3, 2, figsize=(18, 16))
axes[0, 0].plot(monthly_trend.index, monthly_trend.values, marker='o')
axes[0, 0].set_title("Monthly Crime Trends")
axes[0, 0].set_xlabel("Month-Year")
axes[0, 0].set_ylabel("Number of Crimes")
axes[0, 0].tick_params(axis='x', rotation=45)

# Clean lat/lon further to avoid plotting issues
lat_lon = lat_lon[(lat_lon['LAT'] != 0) & (lat_lon['LON'] != 0)]
lat_lon = lat_lon.dropna(subset=['LAT', 'LON'])

if len(lat_lon) > 0:
    sns.kdeplot(
        x=lat_lon['LON'], y=lat_lon['LAT'],
        cmap="Reds", fill=True, ax=axes[0, 1], thresh=0.05
    )
    axes[0, 1].set_title("Crime Density by Location")
else:
    axes[0, 1].text(0.5, 0.5, 'No valid location data', ha='center')


sns.barplot(x=hourly_crime.index, y=hourly_crime.values, ax=axes[1, 0], palette='Blues_d')
axes[1, 0].set_title("Crimes by Hour")

weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
sns.barplot(x=weekday_labels, y=weekday_crime.values, ax=axes[1, 1], palette='Greens_d')
axes[1, 1].set_title("Crimes by Weekday")

sns.barplot(y=top_crimes.index, x=top_crimes.values, ax=axes[2, 0], palette='Reds_d')
axes[2, 0].set_title("Top 15 Most Common Crime Types")

sns.histplot(vict_age_dist, bins=30, kde=True, ax=axes[2, 1], color='purple')
axes[2, 1].set_title("Victim Age Distribution")

plt.tight_layout()
plt.show()

# ARIMA Forecast
register_matplotlib_converters()
monthly_crime = df_clean.groupby('Month_Year').size().sort_index()
monthly_crime.index = pd.to_datetime(monthly_crime.index)
monthly_crime = monthly_crime.asfreq('MS')
adf_result = adfuller(monthly_crime)
if adf_result[1] > 0.05:
    monthly_crime = monthly_crime.diff().dropna()
model = ARIMA(monthly_crime, order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)
plt.figure(figsize=(12, 6))
plt.plot(monthly_crime, label='Observed')
plt.plot(forecast.index, forecast.values, label='Forecast', color='blue', linestyle='--')
plt.title("ARIMA Forecast of Total Monthly Crimes")
plt.xlabel("Date")
plt.ylabel("Number of Crimes")
plt.legend()
plt.tight_layout()
plt.show()

# LSTM for specific crime type
type_monthly = df_clean.groupby(['Month_Year', 'Crm_Cd_Desc']).size().unstack(fill_value=0)
type_monthly.index = pd.to_datetime(type_monthly.index)
type_monthly = type_monthly.asfreq('MS')
example_crime = 'VEHICLE - STOLEN'
crime_series = type_monthly[example_crime].values.reshape(-1, 1)
scaler = MinMaxScaler()
crime_scaled = scaler.fit_transform(crime_series)

def create_sequences(data, seq_len=12):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(crime_scaled)
X = X.reshape((X.shape[0], X.shape[1], 1))
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=1)

input_seq = crime_scaled[-12:].reshape(1, 12, 1)
preds = []
for _ in range(12):
    pred = model.predict(input_seq)[0]
    preds.append(pred)
    input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)

preds = scaler.inverse_transform(preds)
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
