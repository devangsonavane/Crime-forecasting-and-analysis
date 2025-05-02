# 🔍 Crime Forecasting & Analysis using Big Data and Machine Learning

This project provides a complete pipeline for analyzing, visualizing, and forecasting crime trends using historical crime data from Los Angeles. Leveraging techniques in Big Data, time series forecasting (ARIMA, LSTM, Prophet), and geospatial heatmaps, the system aims to assist public safety decision-making with data-driven insights.

---

## 📂 Folder Structure

big-data-project/
├── Data/                         # Contains split CSV files (10000 rows each)
├── Papers/                       # Reference research papers
├── crime\_hotspot\_map.html       # Interactive map output
├── Analysis.png                 # Visual summary or infographic
├── \*.py                         # Python scripts for cleaning, EDA, forecasting
├── .gitignore
└── README.md

---

## 📊 Dataset Used

**Source:** [Los Angeles Crime Dataset (2020 - Present)](https://www.kaggle.com/datasets/nathaniellybrand/los-angeles-crime-dataset-2020-present)

> ❗ The original CSV file (`Crime_Data_from_2020_to_Present.csv`) is over 240MB and not included in this repo.  
> Instead, we provide split CSV files (`split_1.csv`, `split_2.csv`, ...) for easier handling and reproducibility.

---

## 🧠 Models Implemented

| Model   | Purpose                         | Script(s)               |
|---------|----------------------------------|--------------------------|
| **ARIMA**   | Monthly crime trend forecasting | `Crime.py`, `Crime3.py` |
| **LSTM**    | Predict future crime rates      | `Crime.py`, `Crime2.py`, `Crime4.py` |
| **Prophet** | Time series decomposition & forecast | `Crime2.py`, `Crime3.py` |
| **XGBoost** | Crime type classification       | `final_crime_analysis_script.py` |
| **Folium**  | Heatmap for hotspot mapping     | `heatmap.py` |

---

## 📦 Dependencies

Install required Python packages:

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras folium prophet

> ⚠️ If Prophet fails to install, use:

pip install pystan==2.19.1.1
pip install prophet

---

## ⚙️ How to Run the Project

### 1. 🔧 Preprocessing

Run the script to clean and transform raw data:

python Data_cleaning.py

### 2. 📊 Exploratory Data Analysis

Visualize patterns:

python EDA.py

### 3. 📈 Forecast Crime Trends

* ARIMA Model:

python Crime.py

* LSTM Forecast:

python Crime2.py

* Prophet Model:

python Crime3.py

### 4. 🗺️ Generate Crime Heatmap

python heatmap.py
# Output: opens crime_hotspot_map.html

### 5. 🧪 Full Combined Pipeline (EDA + Forecasting + Classification)

python final_crime_analysis_script.py

---

## 🧪 Files Overview

| File                             | Purpose                                             |
| -------------------------------- | --------------------------------------------------- |
| `Data_cleaning.py`               | Clean raw CSV and add time-based features           |
| `EDA.py`                         | Monthly, hourly, weekday crime analysis             |
| `Crime.py`                       | ARIMA + LSTM on single class                        |
| `Crime2.py`, `Crime3.py`         | Prophet-based forecasting                           |
| `Crime4.py`                      | Clean ARIMA + Prophet test                          |
| `Forecasting.py`                 | Generic forecasting logic                           |
| `heatmap.py`                     | Creates a folium map of crime hotspots              |
| `splitter.py`                    | Splits large CSV into smaller chunks                |
| `final_crime_analysis_script.py` | One-click full pipeline (clean → analyze → predict) |

---

## 📚 References

This project is inspired and supported by these key research papers:

1. **Big Data Analytics and Mining for Crime Forecasting** – IEEE Access
2. **Crime Data Analysis Using Pig with Hadoop** – Procedia Computer Science
3. **Crime Analysis Through Machine Learning** – SpringerLink
4. **Using ML Algorithms to Analyze Crime Data** – MLAIJ

Find these in the `Papers/` folder.

---

## ✅ Outcome

By the end of this project, users can:

* Clean and preprocess real-world crime data
* Analyze and visualize crime patterns by time and location
* Forecast future crime trends using time series and deep learning
* Classify crime types based on spatio-temporal features
* Deploy an interactive hotspot map for city-wide crime density

---

## 🤝 Contribution

Feel free to fork this repository, suggest improvements, or create pull requests.
For queries, raise an issue or reach out via GitHub Discussions.

---

## ⚠️ License

This project is released under the MIT License.
Dataset is in the **Public Domain** (via Kaggle) — use it responsibly for research or analysis.

---

## 🏁 Start Now!

git clone https://github.com/devangsonavane/Crime-forecasting-and-analysis.git
cd Crime-forecasting-and-analysis
python final_crime_analysis_script.py

Happy Forecasting! 📈🕵️‍♂️
