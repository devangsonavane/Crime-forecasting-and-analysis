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
