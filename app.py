import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from flask import Flask, request, jsonify, render_template
import joblib
from datetime import date, timedelta, datetime

# Initialize the Flask application
app = Flask(__name__)

# --- Load Models and Supporting Files ---
print("Loading models and feature columns...")
try:
    lgb_model = lgb.Booster(model_file='lgb_model.txt')
    xgb_model = xgb.Booster()
    xgb_model.load_model('xgb_model.json')
    feature_columns = joblib.load('feature_columns.json')
    store_df = pd.read_csv("store.csv")
    print("Models and data loaded successfully.")
except Exception as e:
    print(f"Error loading models or data: {e}")
    # Exit or handle the error appropriately if models can't be loaded
    lgb_model = xgb_model = feature_columns = store_df = None

# --- Feature Engineering Function ---
def feature_engineer(df):
    """Creates new features for the Rossmann dataset on a merged dataframe."""
    # Ensure Date column is in datetime format, but only convert if not already
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Fill NaNs for calculation-dependent columns first
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(df['Date'].dt.month)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(df['Date'].dt.year)
    df['CompetitionDistance'].fillna(0, inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna(0, inplace=True)
    
    # Convert categorical columns to codes
    df['StoreType'] = df['StoreType'].astype('category').cat.codes
    df['Assortment'] = df['Assortment'].astype('category').cat.codes
    # Handle StateHoliday: The incoming JSON has '0', 'a', etc. as strings
    df['StateHoliday'] = df['StateHoliday'].replace({'0': 0, 0: 0, 'a': 1, 'b': 2, 'c': 3}).astype(int)

    # Time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    
    df['CompetitionOpenSince'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpenSince'] = df['CompetitionOpenSince'].apply(lambda x: x if x > 0 else 0)

    df['Promo2Since'] = 12 * (df['Year'] - df['Promo2SinceYear']) + (df['WeekOfYear'] / 4.0 - df['Promo2SinceWeek'] / 4.0)
    df['Promo2Since'] = df['Promo2Since'].apply(lambda x: x if x > 0 else 0)

    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df['MonthStr'] = df['Month'].map(month_map)
    df['IsPromoMonth'] = df.apply(lambda row: 1 if isinstance(row['PromoInterval'], str) and row['MonthStr'] in row['PromoInterval'] else 0, axis=1)
    
    # Drop helper column
    df.drop('MonthStr', axis=1, inplace=True)
    
    return df

# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the main page with the form."""
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast_gui():
    """Handles the prediction request from the HTML form for a 6-week forecast."""
    try:
        store_id = int(request.form['store_id'])
        
        if store_id not in store_df['Store'].unique():
            return render_template('index.html', error=f"Store ID {store_id} not found.")

        # Create DataFrame for the next 6 weeks (42 days)
        today = date.today()
        future_dates = [today + timedelta(days=i) for i in range(42)]
        future_df = pd.DataFrame({'Date': future_dates})
        print(f"Initial future_df['Date'] type: {future_df['Date'].dtype}")
        print(f"Initial future_df['Date'] sample: {future_df['Date'].head().tolist()}")  # Check first few dates
        future_df['Store'] = store_id
        
        # Ensure Date is datetime64[ns]
        future_df['Date'] = pd.to_datetime(future_df['Date'])
        
        # Assumptions for future data
        future_df['Open'] = future_df['Date'].dt.dayofweek.apply(lambda x: 0 if x == 6 else 1)
        future_df['Promo'] = 0  # Assuming no promo for simplicity
        future_df['StateHoliday'] = 0
        future_df['SchoolHoliday'] = 0
        
        # Merge with store_df
        merged_df = pd.merge(future_df, store_df, on='Store', how='left')
        print(f"Merged_df['Date'] type: {merged_df['Date'].dtype}")
        print(f"Merged_df['Date'] sample: {merged_df['Date'].head().tolist()}")  # Check after merge
        print(f"Merged_df columns: {merged_df.columns.tolist()}")  # Check all columns
        
        # Engineer features
        featured_df = feature_engineer(merged_df.copy())
        print(f"Featured_df['Date'] type: {featured_df['Date'].dtype}")
        print(f"Featured_df columns: {featured_df.columns.tolist()}")  # Check after feature engineering
        
        processed_df = featured_df[feature_columns]
        print(f"Processed_df columns: {processed_df.columns.tolist()}")  # Check before prediction
        
        lgb_pred = lgb_model.predict(processed_df)
        xgb_pred = xgb_model.predict(xgb.DMatrix(processed_df))
        ensemble_pred = np.expm1(0.5 * lgb_pred + 0.5 * xgb_pred)
        ensemble_pred[future_df['Open'] == 0] = 0

        predictions = {
            'dates': [d.strftime('%Y-%m-%d (%A)') for d in future_dates],
            'sales': [round(p, 2) for p in ensemble_pred]
        }

        return render_template('index.html', predictions=predictions, store_id=store_id)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """Handles the real-time prediction request for a single data point."""
    if not all([lgb_model, xgb_model, feature_columns is not None, store_df is not None]):
        return jsonify({"error": "Models or necessary data not loaded. Check server logs."}), 500

    try:
        json_data = request.get_json()
        input_df = pd.DataFrame([json_data])
        
        required_keys = ["store", "date", "promo", "state_holiday", "school_holiday", "day_of_week"]
        if not all(key in input_df.columns for key in required_keys):
            return jsonify({"error": f"Missing required keys. Required: {required_keys}"}), 400

        store_id = int(input_df['store'].iloc[0])
        if store_id not in store_df['Store'].unique():
            return jsonify({"error": f"Store ID {store_id} not found."}), 404
            
        input_df.rename(columns={'store': 'Store', 'day_of_week': 'DayOfWeek'}, inplace=True)

        merged_df = pd.merge(input_df, store_df, on='Store', how='left')
        # We need to manually add Open, as it's not in the API input
        merged_df['Open'] = 1  # Assume open for a single prediction
        featured_df = feature_engineer(merged_df.copy())
        processed_df = featured_df[feature_columns]
        
        lgb_pred = lgb_model.predict(processed_df)
        xgb_pred = xgb_model.predict(xgb.DMatrix(processed_df))
        ensemble_pred_log = 0.5 * lgb_pred + 0.5 * xgb_pred
        final_prediction = np.expm1(ensemble_pred_log[0])
        
        return jsonify({'predicted_sales': round(final_prediction, 2)})

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)