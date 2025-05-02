import pandas as pd
import numpy as np
import ace_tools as tools; tools.display_dataframe_to_user(name="Cleaned Crime Data", dataframe=df_clean)

# Load the uploaded CSV file
file_path = "/mnt/data/split_1.csv"
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
