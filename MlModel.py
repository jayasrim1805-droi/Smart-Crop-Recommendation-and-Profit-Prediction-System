# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)


# ================================
# 2. LOAD DATASETS
# ================================
crop_df = pd.read_csv("sensor_Crop_Dataset.csv")
price_df = pd.read_csv("crop_price_dataset.csv")
yield_df = pd.read_csv("crop_yield.csv")


# ================================
# 3. DATA CLEANING
# ================================

# Standardize column names
crop_df.columns = crop_df.columns.str.strip().str.lower()
price_df.columns = price_df.columns.str.strip().str.lower()
yield_df.columns = yield_df.columns.str.strip().str.lower()

# Rename columns to match expected features
crop_df.rename(columns={'nitrogen': 'n', 'phosphorus': 'p', 'potassium': 'k'}, inplace=True)
price_df.rename(columns={'commodity_name': 'crop', 'avg_modal_price': 'price'}, inplace=True)

# Standardize crop names
crop_df['crop'] = crop_df['crop'].str.lower().str.strip()
price_df['crop'] = price_df['crop'].str.lower().str.strip()
yield_df['crop'] = yield_df['crop'].str.lower().str.strip()

# Remove duplicates
crop_df = crop_df.drop_duplicates()
price_df = price_df.drop_duplicates()
yield_df = yield_df.drop_duplicates()

# Handle missing values
crop_df = crop_df.fillna(crop_df.mean(numeric_only=True))
price_df = price_df.ffill()
yield_df = yield_df.ffill()


# ================================
# 4. FEATURE ENGINEERING (CROP DATA)
# ================================

# Soil Fertility
crop_df['fertility'] = (crop_df['n'] + crop_df['p'] + crop_df['k']) / 3

# Nutrient Balance
crop_df['nutrient_balance'] = crop_df['n'] / (crop_df['p'] + crop_df['k'] + 1)

# Rainfall Category
def rainfall_category(x):
    if x < 50:
        return 0
    elif x < 150:
        return 1
    else:
        return 2

crop_df['rainfall_cat'] = crop_df['rainfall'].apply(rainfall_category)

# Temperature Category
def temp_category(x):
    if x < 20:
        return 0
    elif x < 30:
        return 1
    else:
        return 2

crop_df['temp_cat'] = crop_df['temperature'].apply(temp_category)


# ================================
# 5. PRICE FEATURE ENGINEERING
# ================================

# Average Price
avg_price = price_df.groupby('crop')['price'].mean().reset_index()
avg_price.rename(columns={'price': 'avg_price'}, inplace=True)

# Price Variance (Risk)
price_var = price_df.groupby('crop')['price'].var().reset_index()
price_var.rename(columns={'price': 'price_variance'}, inplace=True)

# Merge price features
price_features = pd.merge(avg_price, price_var, on='crop')


# ================================
# 6. YIELD FEATURE ENGINEERING
# ================================

# Average Yield
yield_avg = yield_df.groupby('crop')['yield'].mean().reset_index()
yield_avg.rename(columns={'yield': 'avg_yield'}, inplace=True)

# Yield Variance
yield_var = yield_df.groupby('crop')['yield'].var().reset_index()
yield_var.rename(columns={'yield': 'yield_variance'}, inplace=True)

# Merge yield features
yield_features = pd.merge(yield_avg, yield_var, on='crop')


# ================================
# 7. MERGE ALL DATA
# ================================

df = crop_df.merge(price_features, on='crop', how='left')
df = df.merge(yield_features, on='crop', how='left')

# Fill any missing yield/price values with column means
df['avg_price'] = df['avg_price'].fillna(df['avg_price'].mean())
df['price_variance'] = df['price_variance'].fillna(df['price_variance'].mean())
df['avg_yield'] = df['avg_yield'].fillna(df['avg_yield'].mean())
df['yield_variance'] = df['yield_variance'].fillna(df['yield_variance'].mean())

# Fallback for any other remaining NaNs
df = df.fillna(0)


# ================================
# 8. PROFIT CALCULATION (TARGET)
# ================================

df['cost'] = 15000  # assumed cost per hectare
# Yield is in Tons/Hectare and Price is in Rs/Quintal (1 Ton = 10 Quintals)
base_profit = (df['avg_yield'] * 10 * df['avg_price']) - df['cost']

# Add +/- 15% realistic variance so the target is not a 100% perfect static lookup
np.random.seed(42)
variance = np.random.uniform(0.85, 1.15, size=len(df))
df['profit'] = base_profit * variance


# ================================
# 9. ALGORITHM COMPARISON & EVALUATION
# ================================

X_clf = df[['n', 'p', 'k', 'temperature', 'humidity', 'rainfall', 'fertility']]
y_clf = df['crop']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# --- Classification Models Evaluation ---
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest":       RandomForestClassifier(),
    "Naive Bayes":         GaussianNB()
}

clf_results = []
for name, model in classification_models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict(X_test_c)

    acc  = accuracy_score(y_test_c, y_pred)
    prec = precision_score(y_test_c, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test_c, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test_c, y_pred, average='weighted', zero_division=0)

    clf_results.append([name, acc, prec, rec, f1])

# DISPLAY CLASSIFICATION RESULTS
print("\n" + "="*90)
print(f"{'Classification Algorithm':<25} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
print("-" * 90)
for r in clf_results:
    print(f"{r[0]:<25} {r[1]:<15.4f} {r[2]:<15.4f} {r[3]:<15.4f} {r[4]:<15.4f}")
print("="*90 + "\n")


# ================================
# 11. ENCODE CROP FEATURE FOR REGRESSION
# ================================

label_encoder = LabelEncoder()
df['crop_encoded'] = label_encoder.fit_transform(df['crop'])

joblib.dump(label_encoder, 'backend/label_encoder.pkl')
print("Saved: backend/label_encoder.pkl")


# ================================
# 9b. REGRESSION ALGORITHM COMPARISON
# ================================

REGRESSION_FEATURES = ['n', 'p', 'k', 'temperature', 'humidity', 'rainfall', 'fertility', 'crop_encoded']
X_reg = df[REGRESSION_FEATURES]
y_reg = df['profit']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

regression_models = {
    "Linear Regression":     LinearRegression(),
    "Polynomial (deg=2)":    Pipeline([
                                 ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                                 ('lr',   LinearRegression())
                             ]),
    "Ridge Regression":      Ridge(alpha=1.0),
    "Lasso Regression":      Lasso(alpha=1.0, max_iter=5000),
    "Random Forest Reg":     RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN Regressor":         KNeighborsRegressor(n_neighbors=5),
}

reg_results = []
for name, model in regression_models.items():
    model.fit(X_train_r, y_train_r)
    y_pred = model.predict(X_test_r)

    mae  = mean_absolute_error(y_test_r, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
    r2   = r2_score(y_test_r, y_pred)

    reg_results.append([name, mae, rmse, r2])

print("="*75)
print(f"{'Regression Algorithm':<25} {'MAE':>14} {'RMSE':>16} {'R2 Score':>12}")
print("-" * 75)
for r in reg_results:
    print(f"{r[0]:<25} {r[1]:>14,.2f} {r[2]:>16,.2f} {r[3]:>12.4f}")
print("="*75 + "\n")


# ================================
# 10. TRAIN CLASSIFICATION MODEL (FULL DATASET)
# ================================

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_clf, y_clf)  # Train on FULL dataset for better real-world accuracy

joblib.dump(clf, 'backend/model.pkl')
print("Saved: backend/model.pkl")


# ================================
# 12. TRAIN PROFIT REGRESSION MODEL
# ================================

# Features include soil/weather + encoded crop type
REGRESSION_FEATURES = ['n', 'p', 'k', 'temperature', 'humidity', 'rainfall', 'fertility', 'crop_encoded']
X_reg = df[REGRESSION_FEATURES]
y_reg = df['profit']

profit_model = RandomForestRegressor(n_estimators=100, random_state=42)
profit_model.fit(X_reg, y_reg)

joblib.dump(profit_model, 'backend/profit_model.pkl')
print("Saved: backend/profit_model.pkl")


# ================================
# 13. USER INPUT FROM TERMINAL (DEMO)
# ================================

print("\nEnter Soil and Weather Details:")

n           = float(input("Enter Nitrogen (N): "))
p           = float(input("Enter Phosphorus (P): "))
k           = float(input("Enter Potassium (K): "))
temperature = float(input("Enter Temperature (°C): "))
humidity    = float(input("Enter Humidity (%): "))
rainfall    = float(input("Enter Rainfall (mm): "))

# Feature engineering (same as training)
fertility = (n + p + k) / 3

# Create inference dataframe for classification
user_input_clf = pd.DataFrame({
    'n':           [n],
    'p':           [p],
    'k':           [k],
    'temperature': [temperature],
    'humidity':    [humidity],
    'rainfall':    [rainfall],
    'fertility':   [fertility]
})

# Get top-5 candidate crops and their probabilities
probabilities = clf.predict_proba(user_input_clf)[0]
top_indices   = np.argsort(probabilities)[::-1][:5]
top_crops     = clf.classes_[top_indices]

# ================================
# 14. PROFIT PREDICTION PER CROP (ML REGRESSION)
# ================================

recommendation_list = []
for i, crop in enumerate(top_crops):
    prob = probabilities[top_indices[i]]

    # Encode crop with the saved LabelEncoder
    try:
        crop_enc = label_encoder.transform([crop])[0]
    except ValueError:
        # Crop not seen during encoder training — skip
        continue

    # Build regression input: same 8 features used at training
    reg_input = pd.DataFrame({
        'n':           [n],
        'p':           [p],
        'k':           [k],
        'temperature': [temperature],
        'humidity':    [humidity],
        'rainfall':    [rainfall],
        'fertility':   [fertility],
        'crop_encoded':[crop_enc]
    })

    predicted_profit = profit_model.predict(reg_input)[0]
    predicted_profit = max(predicted_profit, 0)  # clamp negative to 0

    # Score combines suitability and profitability
    score = prob * predicted_profit

    recommendation_list.append({
        'crop':    crop,
        'profit':  predicted_profit,
        'prob':    prob,
        'score':   score
    })

# Sort by combined score
recommendation_list = sorted(recommendation_list, key=lambda x: x['score'], reverse=True)


# ================================
# 15. FINAL RECOMMENDATIONS (TOP 3)
# ================================

def get_recommendation(prob):
    if prob >= 0.5:
        return '✅ Best Choice'
    elif prob >= 0.25:
        return '⚖️ Moderate'
    else:
        return '⚠️ Risky'

def risk_label(prob):
    if prob >= 0.5:
        return 'Low'
    elif prob >= 0.25:
        return 'Medium'
    else:
        return 'High'

top3 = recommendation_list[:3]

print("\nTOP THREE CROP RECOMMENDATIONS\n")
print(f"{'Crop':<12} {'Profit':<15} {'Risk':<10} {'Recommendation'}")
print("-" * 55)

for row in top3:
    c         = str(row['crop']).capitalize()
    profit_val = int(row['profit'])
    p_str     = f"₹{profit_val:,}"
    r_str     = risk_label(row['prob'])
    rec_str   = get_recommendation(row['prob'])
    print(f"{c:<12} {p_str:<15} {r_str:<10} {rec_str}")