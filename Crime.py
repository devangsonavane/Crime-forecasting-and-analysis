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

# --------------------------------------------------
# Data Loading and Cleaning Function
# --------------------------------------------------
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads and cleans the crime data CSV file.
    
    - Reads CSV.
    - Strips whitespace and replaces spaces in column names.
    - Converts date columns and drops rows with missing dates.
    - Fills missing values for categorical and numerical columns.
    - Casts selected columns to category.
    - Creates additional time-based features.
    """
    df = pd.read_csv(file_path)
    df_clean = df.copy()
    
    # Clean column names
    df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_')
    
    # Convert date columns
    df_clean['DATE_OCC'] = pd.to_datetime(df_clean['DATE_OCC'], 
                                            format='%m/%d/%Y %I:%M:%S %p', 
                                            errors='coerce')
    df_clean['Date_Rptd'] = pd.to_datetime(df_clean['Date_Rptd'], 
                                           format='%m/%d/%Y %I:%M:%S %p', 
                                           errors='coerce')
    df_clean = df_clean.dropna(subset=['DATE_OCC', 'Date_Rptd'])
    
    # Fill missing values
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    df_clean[cat_cols] = df_clean[cat_cols].fillna('Unknown')
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(-1)
    
    # Cast to category for select columns
    category_candidates = ['Vict_Sex', 'Vict_Descent', 'Status', 
                           'Status_Desc', 'Part_1-2', 'AREA_NAME']
    for col in category_candidates:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')
    
    # Create time-based features
    df_clean['Year'] = df_clean['DATE_OCC'].dt.year
    df_clean['Month'] = df_clean['DATE_OCC'].dt.month
    df_clean['Day'] = df_clean['DATE_OCC'].dt.day
    if 'TIME_OCC' in df_clean.columns:
        # Assumes TIME_OCC is in HHMM format as integer or string convertible to int
        df_clean['Hour'] = df_clean['TIME_OCC'].astype(int) // 100
    df_clean['Weekday'] = df_clean['DATE_OCC'].dt.dayofweek
    df_clean['Month_Year'] = df_clean['DATE_OCC'].dt.to_period('M').astype(str)
    
    if 'Crm_Cd_Desc' in df_clean.columns:
        df_clean['Crm_Cd_Desc'] = df_clean['Crm_Cd_Desc'].str.upper().str.strip()
    
    return df_clean

# --------------------------------------------------
# Visualization Function
# --------------------------------------------------
def plot_visualizations(df: pd.DataFrame) -> None:
    """
    Plots various visualizations:
    - Monthly crime trends.
    - Crime density using valid lat/lon.
    - Crimes by hour and weekday.
    - Top 15 most common crime types.
    - Victim age distribution.
    """
    # Set plot style
    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (12, 6)
    
    # Compute aggregates
    monthly_trend = df['Month_Year'].value_counts().sort_index()
    hourly_crime = df['Hour'].value_counts().sort_index()
    weekday_crime = df['Weekday'].value_counts().sort_index()
    top_crimes = df['Crm_Cd_Desc'].value_counts().head(15)
    vict_age_dist = df[df['Vict_Age'] > 0]['Vict_Age']
    
    # Filter valid location data
    lat_lon = df[(df['LAT'] != -1) & (df['LON'] != -1)]
    lat_lon = lat_lon[(lat_lon['LAT'] != 0) & (lat_lon['LON'] != 0)]
    lat_lon = lat_lon.dropna(subset=['LAT', 'LON'])
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    
    # Monthly trends
    axes[0, 0].plot(monthly_trend.index, monthly_trend.values, marker='o')
    axes[0, 0].set_title("Monthly Crime Trends")
    axes[0, 0].set_xlabel("Month-Year")
    axes[0, 0].set_ylabel("Number of Crimes")
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Crime density map
    if not lat_lon.empty:
        sns.kdeplot(x=lat_lon['LON'], y=lat_lon['LAT'], cmap="Reds", fill=True, 
                    ax=axes[0, 1], thresh=0.05)
        axes[0, 1].set_title("Crime Density by Location")
    else:
        axes[0, 1].text(0.5, 0.5, 'No valid location data', ha='center')
    
    # Crimes by hour
    sns.barplot(x=hourly_crime.index, y=hourly_crime.values, ax=axes[1, 0], palette='Blues_d')
    axes[1, 0].set_title("Crimes by Hour")
    
    # Crimes by weekday
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sns.barplot(x=weekday_labels, y=weekday_crime.values, ax=axes[1, 1], palette='Greens_d')
    axes[1, 1].set_title("Crimes by Weekday")
    
    # Top 15 crime types
    sns.barplot(y=top_crimes.index, x=top_crimes.values, ax=axes[2, 0], palette='Reds_d')
    axes[2, 0].set_title("Top 15 Most Common Crime Types")
    
    # Victim age distribution
    sns.histplot(vict_age_dist, bins=30, kde=True, ax=axes[2, 1], color='purple')
    axes[2, 1].set_title("Victim Age Distribution")
    
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# ARIMA Forecast Function
# --------------------------------------------------
def forecast_arima(df: pd.DataFrame, forecast_steps: int = 12) -> None:
    """
    Forecasts total monthly crimes using an ARIMA model.
    Checks for stationarity and differences the series if needed.
    Plots the observed and forecasted values.
    """
    register_matplotlib_converters()
    # Group by month-year and convert index
    monthly_crime = df.groupby('Month_Year').size().sort_index()
    monthly_crime.index = pd.to_datetime(monthly_crime.index)
    monthly_crime = monthly_crime.asfreq('MS')
    
    # Check stationarity
    adf_result = adfuller(monthly_crime.dropna())
    if adf_result[1] > 0.05:
        # Non-stationary: difference the series and note that ARIMA order reflects that differencing
        monthly_crime = monthly_crime.diff().dropna()
        arima_order = (1, 0, 1)
    else:
        arima_order = (1, 0, 1)
    
    # Fit ARIMA model and forecast
    arima_model = ARIMA(monthly_crime, order=arima_order)
    model_fit = arima_model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Plot observed data and forecast
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_crime.index, monthly_crime.values, label='Observed')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='blue', linestyle='--')
    plt.title("ARIMA Forecast of Total Monthly Crimes")
    plt.xlabel("Date")
    plt.ylabel("Number of Crimes")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Helper function to create sequences for LSTM
# --------------------------------------------------
def create_sequences(data: np.ndarray, seq_len: int = 12):
    """
    Creates input-output sequences from the data for LSTM.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# --------------------------------------------------
# LSTM Forecast Function for a Specific Crime Type
# --------------------------------------------------
def forecast_lstm(df: pd.DataFrame, crime_type: str = 'VEHICLE - STOLEN', 
                  forecast_steps: int = 12, seq_len: int = 12, epochs: int = 500) -> None:
    """
    Forecasts the crime counts for a specific crime type using an LSTM model.
    Prepares the time series, scales the data, creates sequences,
    trains the model, and iteratively forecasts the specified steps.
    """
    # Pivot data: rows as Month_Year and columns as crime types
    type_monthly = df.groupby(['Month_Year', 'Crm_Cd_Desc']).size().unstack(fill_value=0)
    type_monthly.index = pd.to_datetime(type_monthly.index)
    type_monthly = type_monthly.asfreq('MS')
    
    if crime_type not in type_monthly.columns:
        raise ValueError(f"Crime type '{crime_type}' not found in data.")
    
    # Prepare the time series for the selected crime type
    crime_series = type_monthly[crime_type].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    crime_scaled = scaler.fit_transform(crime_series)
    
    # Create sequences for training
    X, y = create_sequences(crime_scaled, seq_len=seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Build and train LSTM model
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X, y, epochs=epochs, verbose=1)
    
    # Iterative forecasting for forecast_steps months
    input_seq = crime_scaled[-seq_len:].reshape(1, seq_len, 1)
    preds = []
    for _ in range(forecast_steps):
        pred = lstm_model.predict(input_seq)[0]
        preds.append(pred)
        # Append the new prediction and slide the window
        input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)
    
    # Inverse transform predictions
    preds_inverse = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    
    # Create forecast index for plotting
    forecast_index = pd.date_range(start=type_monthly.index[-1] + pd.DateOffset(months=1), 
                                   periods=forecast_steps, freq='MS')
    
    # Plot observed and forecasted crime counts
    plt.figure(figsize=(12, 6))
    plt.plot(type_monthly.index, crime_series, label='Observed')
    plt.plot(forecast_index, preds_inverse, label='Forecast', linestyle='--', color='orange')
    plt.title(f"LSTM Forecast for '{crime_type}' Crimes")
    plt.xlabel("Date")
    plt.ylabel("Number of Crimes")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if __name__ == "__main__":
    # Adjust the file path as needed
    file_path = "E:/Programs/SEM 6/BDA/Project/Crime_Data_from_2020_to_Present.csv"
    df_clean = load_and_clean_data(file_path)
    
    # Plot visualizations
    plot_visualizations(df_clean)
    
    # Forecast using ARIMA
    forecast_arima(df_clean, forecast_steps=12)
    
    # Forecast using LSTM for a specific crime type
    forecast_lstm(df_clean, crime_type='VEHICLE - STOLEN', forecast_steps=12, seq_len=12, epochs=50)
