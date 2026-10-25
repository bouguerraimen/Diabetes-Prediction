from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.exceptions import NotFittedError
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

N_FEATURES = 8  

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    features = data.get("features")

    if not isinstance(features, list) or len(features) != N_FEATURES:
        return jsonify({"error": f"Il faut exactement {N_FEATURES} valeurs numériques."}), 400

    try:
        X = np.array(features, dtype=float).reshape(1, -1)
    except (TypeError, ValueError):
        return jsonify({"error": "Valeurs invalides."}), 400

    if not np.isfinite(X).all() or (X < 0).any():
        return jsonify({"error": "Les valeurs doivent être des nombres positifs."}), 400

    if scaler is not None:
        X = scaler.transform(X)

    pred = int(model.predict(X)[0])

    confidence = None
    try:
        confidence = float(np.max(model.predict_proba(X)[0]))
    except (AttributeError, NotFittedError):
        pass

    label = "diabetic" if pred == 1 else "not diabetic"
    return jsonify({"prediction": pred, "label": label, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)