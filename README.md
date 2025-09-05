# SmartStock-An Inventory Management System

## Problem Statement
Businesses often struggle with inventory management. Overstocking leads to tied-up capital and waste, while understocking leads to lost sales and unhappy customers. Manual methods or basic systems fail to account for real-time demand, pricing changes, seasonality, and other factors like weather or holidays.

## Solution Proposed
SmartStock is an inventory optimization web app.
It uses machine learning to forecast product demand based on factors like store, product, price, discount, seasonality, weather, and holidays. The app calculates important inventory metrics such as safety stock, reorder points, turnover ratios, and stockout risk. By providing real-time, data-backed recommendations, SmartStock helps businesses:

1]Reduce excess inventory

2]Prevent stockouts

3]Improve cash flow

4]Make better purchasing decisions

5]Built with Flask and integrated ML models, SmartStock is user-friendly and can be easily adapted for different businesses and industries.


## Dataset Used
Dataset link:https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset

## Tech Stack Used
Flask
Python
Scikit-learn
numpy
pandas
joblib
html
css


## How to Run
**Step 1. Clone the repository.**
```powershell
git clone https://https://github.com/harshvgangawane/SmartStock.git
cd "Customer Churn Prediction"
```
**Step 2. (Optional) Create a virtual environment.**
```powershell
python -m venv venv
.\venv\Scripts\activate
```
**Step 3. Install the requirements**
```powershell
pip install -r requirements.txt
```
**Step 4. Run the Streamlit app**
```powershell
streamlit run app.py
```

## Project Architecture
-Data Layer: CSV files and preprocessing notebooks (likely in notebook/)
-Model Layer: Trained regression model, scaler, and label encoders stored in models/
-Interface Layer: Flask web app (app.py) serving HTML templates (templates/)
-Presentation Layer: HTML (Jinja2 templates), CSS, and JavaScript in templates/ and static/

## Notebooks
-pipeline_1.ipynb: Exploratory Data Analysis (EDA)
-pipeline_2.ipynb: Feature Engineering, Feature Selection, and Model Training/Validation

## Models Used
- RandomForestRegressor
- Feature scaling with MinMaxScaler
- Label encoding for categorical features

## Conclusion
SmartStock shows how combining machine learning with a simple web interface can solve a real business problem — inventory management. By predicting demand and suggesting key inventory metrics like safety stock and reorder points, it helps businesses avoid both overstocking and stockouts.

