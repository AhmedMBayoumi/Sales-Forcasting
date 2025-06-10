# Rossmann Store Sales Forecasting

This project provides a machine learning-based solution to forecast daily sales for Rossmann stores. It includes a trained model ensemble (LightGBM and XGBoost) and a Flask web server that exposes two endpoints: a simple user interface for generating 6-week forecasts and a REST API for real-time, single-day predictions.

## Overview

The core of this project is an ensemble model that predicts store sales based on historical data and store-specific features. The notebook (`data_scientist_case_study_test.ipynb`) covers the end-to-end process of data loading, feature engineering, model training, and evaluation.

The Flask application (`app.py`) serves as the deployment vehicle for these models.

## Features

- **Dual-Model Ensemble**: Combines the strengths of LightGBM and XGBoost for robust and accurate predictions.
- **Web User Interface**: A simple HTML page (`/forecast`) for non-technical users to get a 6-week sales forecast for any store.
- **REST API**: A real-time prediction endpoint (`/api/predict`) that accepts JSON input and returns a single day's sales forecast.
- **Comprehensive Feature Engineering**: The model leverages time-based, promotional, and competitive features to improve accuracy.

## Setup and Installation

Follow these steps to set up the project and run the server locally.

### 1. Prerequisites

- Python 3.8+
- pip (Python package installer)

### 2. Required Files

Before running the application, ensure you have all the necessary files in the root directory of the project:

- `app.py`: The Flask server script.
- `requirements.txt`: A list of required Python libraries.
- `store.csv`: The dataset containing store information.
- `lgb_model.txt`: The trained LightGBM model.
- `feature_columns.json`: The list of feature columns used for training.
- `templates/index.html`: The HTML template for the GUI.

**Important Note on XGBoost Model:**
The trained XGBoost model file (`xgb_model.json`) was too large to be uploaded to this GitHub repository. You must download it separately from the following link and place it in the root directory of the project.

- **Download Link**: [xgb_model.json on Google Drive](https://drive.google.com/file/d/1qKYrY_EBVSgRXQ7JyVE5BiKjv-HNpmDr/view?usp=sharing)

### 3. Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AhmedMBayoumi/Sales-Forcasting.git](https://github.com/AhmedMBayoumi/Sales-Forcasting.git)
    cd Sales-Forcasting
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 4. Running the Server

Once all the files are in place and dependencies are installed, you can start the Flask server:

```bash
python app.py
```

The server will start and be accessible at `http://127.0.0.1:5000`.

## API Usage

The application provides two ways to get forecasts:

### 1. Web Interface (for 6-week forecast)

-   Open your web browser and navigate to `http://127.0.0.1:5000/`.
-   Enter a valid Store ID into the input field and click "Get Forecast".
-   The page will display a table with the forecasted sales for the next 42 days.

### 2. REST API (for real-time single prediction)

You can send a `POST` request with JSON data to the `/api/predict` endpoint.

-   **Endpoint**: `/api/predict`
-   **Method**: `POST`
-   **Headers**: `Content-Type: application/json`
-   **Body (Example)**:
    ```json
    {
        "store": 215,
        "date": "2015-09-01",
        "promo": 1,
        "state_holiday": "0",
        "school_holiday": 0,
        "day_of_week": 2
    }
    ```

-   **Example with `curl`**:
    ```bash
    curl -X POST [http://127.0.0.1:5000/api/predict](http://127.0.0.1:5000/api/predict) \
    -H "Content-Type: application/json" \
    -d '{
        "store": 215,
        "date": "2015-09-01",
        "promo": 1,
        "state_holiday": "0",
        "school_holiday": 0,
        "day_of_week": 2
    }'
    ```

-   **Success Response**:
    ```json
    {
      "predicted_sales": 4586.42 
    }
    
