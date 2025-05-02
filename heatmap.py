# Re-import necessary libraries
import pandas as pd
import folium
from folium.plugins import HeatMap

# Reload the dataset
df = pd.read_csv("E:\Programs\SEM 6\BDA\Project\Crime_Data_from_2020_to_Present.csv")

# Create a base map centered around the dataset's average location
crime_map = folium.Map(location=[df["LAT"].mean(), df["LON"].mean()], zoom_start=10)

# Prepare heatmap data
heat_data = list(zip(df["LAT"], df["LON"]))

# Add heatmap layer
HeatMap(heat_data, radius=8).add_to(crime_map)

# Save and provide the heatmap file
heatmap_path = "crime_hotspot_map.html"
crime_map.save(heatmap_path)

heatmap_path
