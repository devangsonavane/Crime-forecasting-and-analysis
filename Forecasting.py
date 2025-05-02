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
