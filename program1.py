
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load dataset
file_path = "E:\Programs\SEM 6\BDA\Project\Crime_Data_from_2020_to_Present.csv"  # Ensure the file is in the same directory
df = pd.read_csv(file_path)

# Convert date columns to datetime format
df["Date Rptd"] = pd.to_datetime(df["Date Rptd"], errors='coerce')
df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors='coerce')

# Extract time-based features
df["Hour"] = df["TIME OCC"] // 100  # Convert HHMM to just HH
df["Weekday"] = df["DATE OCC"].dt.weekday  # Monday=0, Sunday=6
df["Month"] = df["DATE OCC"].dt.month

# Define crime category mapping
crime_mapping = {
    "THEFT": "Theft & Larceny",
    "LARCENY": "Theft & Larceny",
    "SHOPLIFTING": "Theft & Larceny",
    "IDENTITY THEFT": "Theft & Larceny",
    "BIKE": "Theft & Larceny",
    "BURGLARY": "Burglary & Trespassing",
    "TRESPASSING": "Burglary & Trespassing",
    "ROBBERY": "Robbery & Armed Crime",
    "BRANDISH WEAPON": "Robbery & Armed Crime",
    "ASSAULT": "Assault & Battery",
    "BATTERY": "Assault & Battery",
    "THREATS": "Assault & Battery",
    "VEHICLE - STOLEN": "Vehicle Crimes",
    "THEFT FROM MOTOR VEHICLE": "Vehicle Crimes",
    "VANDALISM": "Vandalism",
}

# Apply mapping function
def categorize_crime(crime_desc):
    for keyword, category in crime_mapping.items():
        if keyword in str(crime_desc).upper():
            return category
    return "Other Crimes"  # Default category

df["Crime Category"] = df["Crm Cd Desc"].apply(categorize_crime)

# Encode the crime categories
le_crime_category = LabelEncoder()
df["Crime Category Encoded"] = le_crime_category.fit_transform(df["Crime Category"])

# Define features and target variable
features = ["Hour", "Weekday", "Month", "AREA", "Vict Age", "Premis Cd", "Weapon Used Cd", "LAT", "LON"]
target = "Crime Category Encoded"

# Prepare dataset
X = df[features].fillna(-1)
y = df[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, zero_division=0)

# Print results
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_rep)



