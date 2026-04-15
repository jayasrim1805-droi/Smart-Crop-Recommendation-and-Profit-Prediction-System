# 🌾 Smart Crop Recommendation & Profit Prediction System

A sophisticated machine learning platform that empowers farmers by recommending the most suitable crops based on soil and weather conditions, while simultaneously predicting potential profits and assessing financial risks.

---

## 🚀 Overview

This project implements a full-stack predictive maintenance and agricultural optimization tool. By leveraging historical agricultural data, the system provides high-precision crop recommendations and realistic profit forecasts.

### Key Features
- **Intelligent Crop Selection**: Recommends the top 3 most suitable crops based on NPK values, temperature, humidity, and rainfall.
- **Profit Forecasting**: Estimates the expected profit per hectare using a specialized regression model.
- **Risk Assessment**: Categorizes recommendations into *Low*, *Medium*, or *High* risk based on model probability.
- **Interactive UI**: A clean, responsive dashboard for real-time analysis.

---

## 🧠 Machine Learning Model: Random Forest

The core of this system is powered by the **Random Forest** algorithm, which is used for both Classification and Regression tasks.

### Why Random Forest?
- **Robustness**: Handles large datasets with high dimensionality effectively.
- **Non-Linear Relationships**: Excellent at capturing complex interactions between soil nutrients and weather patterns.
- **Prevention of Overfitting**: By using an ensemble of decision trees (Bagging), it provides much better generalization than a single decision tree.
- **Feature Importance**: It naturally handles the varying impact of different nutrients (N, P, K) on different crop types.

### Implementation Details
1.  **RandomForestClassifier**: Used to predict the probability of suitability for various crops.
2.  **RandomForestRegressor**: Specifically trained to predict the profit value for a given crop and environmental context.
3.  **Performance**: The models achieve high accuracy and low MAE (Mean Absolute Error) by training on a combined dataset of soil sensors, historical market prices, and regional yields.

---

## 📂 Directory Structure

```text
.
├── MlModel.py              # ML Pipeline (Data cleaning, Feature Engineering, Training)
├── backend/                # Flask Server
│   ├── app.py              # API Endpoints & Model Serving
│   ├── model.pkl           # Saved Classifier
│   ├── profit_model.pkl    # Saved Regressor
│   └── requirements.txt    # Python Dependencies
├── frontend/               # Interactive Web Interface
│   ├── index.html
│   ├── style.css
│   └── script.js
└── datasets/               # CSV Data (sensor, price, yield)
```

---

## 🛠️ Installation & Setup

### 1. Prerequisite
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
Navigate to the root directory and install the required packages:
```bash
pip install -r backend/requirements.txt
```

---

## 🏃 How to Run

### Step 1: Train the Models
Run the ML pipeline to process the data and generate the model files. This will also show an algorithm comparison table in your terminal.
```bash
python3 MlModel.py
```

### Step 2: Start the Backend Server
Launch the Flask application:
```bash
cd backend
python3 app.py
```

### Step 3: Access the Application
Open your web browser and navigate to:
`http://localhost:8080`

---

## 📊 Technical Workflow
1.  **Data Ingestion**: Merges sensor data, commodity prices, and yield statistics.
2.  **Feature Engineering**: Calculates soil fertility index and cost-to-profit ratios.
3.  **Inference**:
    -   The system predicts crop probabilities.
    -   It calculates a "Suitability Score" by combining the probability with the predicted profit.
    -   Displays the top 3 results ordered by this refined score.

---

> [!TIP]
> **Performance Optimization**: For the best results, ensure your inputs (Nitrogen, Phosphorus, Potassium) are within realistic agricultural ranges for your region.
