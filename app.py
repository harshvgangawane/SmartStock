from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)


reg_model = joblib.load('models/reg_model.pkl')
sc = joblib.load('models/minmax.pkl')
label_encoders = joblib.load('models/label_encoder.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        input_data = {
            'Date': request.form['date'],
            'Store ID': request.form['Store ID'],
            'Product ID': request.form['Product ID'],
            'Category': request.form['Category'],
            'Region': request.form['Region'],
            'Inventory Level': float(request.form['Inventory Level']),
            'Units Ordered': float(request.form['Units Ordered']),
            'Price': float(request.form['Price']),
            'Discount': float(request.form['Discount']),
            'Weather Condition': request.form['Weather Condition'],
            'Holiday': 1 if request.form['Holiday'] == 'Yes' else 0,
            'Seasonality': request.form['Seasonality']
        }
        
        
        date = datetime.strptime(input_data['Date'], '%Y-%m-%d')
        day = date.day
        month = date.month
        year = date.year
        
        
        df = pd.DataFrame([{
            'Category': input_data['Category'],
            'Day': day,
            'Discount': input_data['Discount'],
            'Holiday': input_data['Holiday'],
            'Inventory Level': input_data['Inventory Level'],
            'Month': month,
            'Price': input_data['Price'],
            'Product ID': input_data['Product ID'],
            'Region': input_data['Region'],
            'Seasonality': input_data['Seasonality'],
            'Store ID': input_data['Store ID'],
            'Units Ordered': input_data['Units Ordered'],
            'Weather Condition': input_data['Weather Condition'],
            'Year': year,
            'inventory_turnover_ratio': 0,  
            'lag_14': 0,
            'lag_7': 0,
            'order_to_sales_ratio': 0,  
            'price_discount_interaction': 0,  
            'rolling_mean_7': 0
        }])
        
        cat_features = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']
        
        
        for col in cat_features:
            if col in label_encoders:
                le = label_encoders[col]
                print(f"[DEBUG] Column: {col}, Value(s): {df[col].tolist()}, Classes: {le.classes_}")
                try:
                    df[col] = le.transform(df[col])
                except ValueError as e:
                    print(f"[ERROR] Label encoder for {col} failed: {e}")
                    # Handle unknown categories
                    df[col] = 0
            else:
                print(f"[WARNING] No label encoder found for {col}")
        
    
        df['rolling_mean_7'] = 0
        df['inventory_turnover_ratio'] = df['Units Ordered'] / (df['Inventory Level'] + 1e-6)
        df['order_to_sales_ratio'] = df['Units Ordered'] / (df['Units Ordered'] + 1e-6)
        df['price_discount_interaction'] = df['Price'] * df['Discount']
        df['lag_7'] = 0
        df['lag_14'] = 0
        
    
        print(f"[DEBUG] DataFrame columns before scaling: {df.columns.tolist()}")
        print(f"[DEBUG] DataFrame shape before scaling: {df.shape}")
        print(f"[DEBUG] Sample values: {df.iloc[0].to_dict()}")
        
        
        scale_cols = ['Inventory Level', 'Units Ordered', 'Price', 'Discount',
                      'rolling_mean_7', 'inventory_turnover_ratio',
                      'order_to_sales_ratio', 'price_discount_interaction',
                      'lag_7', 'lag_14']
        
        
        if hasattr(sc, 'feature_names_in_'):
            print(f"[DEBUG] Scaler expects: {sc.feature_names_in_}")
            scaler_expected_cols = sc.feature_names_in_
        else:
            print(f"[DEBUG] Scaler doesn't have feature names, using predefined columns")
            scaler_expected_cols = scale_cols
        
        
        df_to_scale = df[scaler_expected_cols].copy()
        print(f"[DEBUG] Columns being scaled: {df_to_scale.columns.tolist()}")
        print(f"[DEBUG] Shape of data being scaled: {df_to_scale.shape}")
        
        
        try:
            scaled_values = sc.transform(df_to_scale)
            print(f"[DEBUG] Scaling successful, shape: {scaled_values.shape}")
            
            
            scaled_df = pd.DataFrame(scaled_values, columns=scaler_expected_cols, index=df.index)
            
            
            df[scaler_expected_cols] = scaled_df
            
        except Exception as e:
            print(f"[ERROR] Scaling failed: {e}")
            
            pass
        
    
        expected_columns = reg_model.feature_names_in_
        
        # Check if all expected columns exist
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)
        
        if missing_cols:
            print(f"[ERROR] Missing columns: {missing_cols}")
        if extra_cols:
            print(f"[WARNING] Extra columns: {extra_cols}")
        
        
        df_final = df[expected_columns]
        
        print(f"[DEBUG] Final DataFrame shape: {df_final.shape}")
        print(f"[DEBUG] Final DataFrame columns: {df_final.columns.tolist()}")
        print(f"[DEBUG] Model expects: {expected_columns.tolist()}")
        print(f"[DEBUG] Columns match: {df_final.columns.tolist() == expected_columns.tolist()}")
        
        # Make prediction
        prediction = reg_model.predict(df_final)[0]
        prediction = round(prediction, 2)
        
        rolling_mean_7 = 0
        lag_7 = prediction * 0.9
        lag_14 = prediction * 0.85
        forecasted_demand = round(0.5 * prediction + 0.3 * lag_7 + 0.2 * lag_14, 2)
        std_estimate = np.std([prediction, lag_7, lag_14])
        safety_stock = round(1.65 * std_estimate, 2)
        reorder_point = round(forecasted_demand + safety_stock, 2)
        inventory_level= input_data['Inventory Level']
        stockout_risk_score = round(forecasted_demand / (inventory_level + 1e-6), 2)

        
        
        
        return render_template('inventory.html',
            predicted_text=prediction,
            units_sold=prediction,
            inventory_turnover_ratio=round(df['inventory_turnover_ratio'].values[0], 4),
            price_discount_interaction=round(df['price_discount_interaction'].values[0], 4),
            rolling_mean_7=rolling_mean_7,
            seasonality=input_data['Seasonality'],
            holiday=input_data['Holiday'],
            
           
        )

@app.route('/optimize',methods=['GET','POST'])
def optimize():
    units_sold = float(request.form['units_sold'])
    price_discount_interaction = float(request.form['price_discount_interaction'])
    Seasonality = request.form['Seasonality']
    inventory_turnover_ratio = float(request.form['inventory_turnover_ratio'])
    rolling_mean_7 = float(request.form['rolling_mean_7'])
    holiday = int(request.form['holiday'])

    # Example optimization calculation
    forecasted_demand = round(units_sold * (1.05 if Seasonality == 'High' else 1.0), 2)
    safety_stock = round(forecasted_demand * 0.2, 2)
    reorder_point = round(forecasted_demand + safety_stock, 2)
    stockout_risk_score = round((1 - inventory_turnover_ratio) * 10, 2)

    return render_template('inventory.html',
                           units_sold=units_sold,
                           price_discount_interaction=price_discount_interaction,
                           inventory_turnover_ratio=inventory_turnover_ratio,
                           rolling_mean_7=rolling_mean_7,
                           Seasonality=Seasonality,
                           holiday=holiday,
                           forecasted_demand=forecasted_demand,
                           safety_stock=safety_stock,
                           reorder_point=reorder_point,
                           stockout_risk_score=stockout_risk_score)

@app.route('/Dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
    
    