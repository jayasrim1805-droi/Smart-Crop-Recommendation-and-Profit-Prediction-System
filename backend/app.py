import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    return app.send_static_file('index.html')


# ================================
# MODEL LOADING
# ================================
clf          = None
profit_model = None
label_encoder = None

def _load(path_from_root, path_local):
    """Try loading from root path first, then local path."""
    if os.path.exists(path_from_root):
        return joblib.load(path_from_root)
    return joblib.load(path_local)

try:
    clf           = _load('backend/model.pkl',         'model.pkl')
    profit_model  = _load('backend/profit_model.pkl',  'profit_model.pkl')
    label_encoder = _load('backend/label_encoder.pkl', 'label_encoder.pkl')
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")


# ================================
# HELPERS
# ================================

def _risk_label(prob: float) -> str:
    if prob >= 0.5:
        return 'low'
    elif prob >= 0.25:
        return 'medium'
    else:
        return 'high'

def _recommendation(risk: str) -> str:
    return {'low': 'Best Choice', 'medium': 'Moderate', 'high': 'Risky'}.get(risk, 'Moderate')


# ================================
# /predict ENDPOINT
# ================================

@app.route('/predict', methods=['POST'])
def predict():
    if clf is None or profit_model is None or label_encoder is None:
        return jsonify({"error": "Models not loaded on server."}), 500

    try:
        data = request.json

        # Extract features
        n           = float(data.get('n', 0))
        p           = float(data.get('p', 0))
        k           = float(data.get('k', 0))
        temperature = float(data.get('temperature', 0))
        humidity    = float(data.get('humidity', 0))
        rainfall    = float(data.get('rainfall', 0))

        # Derived feature
        fertility = (n + p + k) / 3

        # Classification input (7 features, same as training)
        clf_input = pd.DataFrame({
            'n':           [n],
            'p':           [p],
            'k':           [k],
            'temperature': [temperature],
            'humidity':    [humidity],
            'rainfall':    [rainfall],
            'fertility':   [fertility]
        })

        # Get top-3 crops with their probabilities
        probabilities = clf.predict_proba(clf_input)[0]
        top3_idx      = np.argsort(probabilities)[::-1][:3]
        top3_crops    = clf.classes_[top3_idx]

        # Build results using regression-predicted profit per crop
        results = []
        for idx, crop in zip(top3_idx, top3_crops):
            prob = float(probabilities[idx])

            # Encode crop name to integer
            try:
                crop_enc = int(label_encoder.transform([crop])[0])
            except ValueError:
                # Crop unseen during encoding — skip gracefully
                continue

            # Regression input: 8 features
            reg_input = pd.DataFrame({
                'n':            [n],
                'p':            [p],
                'k':            [k],
                'temperature':  [temperature],
                'humidity':     [humidity],
                'rainfall':     [rainfall],
                'fertility':    [fertility],
                'crop_encoded': [crop_enc]
            })

            predicted_profit = float(profit_model.predict(reg_input)[0])
            predicted_profit = max(predicted_profit, 0)  # clamp negatives to 0

            score = prob * predicted_profit

            risk = _risk_label(prob)
            results.append({
                'crop':           str(crop).capitalize(),
                'profit':         int(predicted_profit),
                'risk':           risk,
                'recommendation': _recommendation(risk),
                '_score':         score   # internal sort key, stripped below
            })

        # Sort by combined score descending
        results.sort(key=lambda x: x['_score'], reverse=True)

        # Remove internal sort key before returning
        for r in results:
            r.pop('_score', None)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
