"""Inference entry script for the Azure ML managed online endpoint.

Azure ML calls init() once on container startup and run() for every request.
"""
import json
import os

import joblib
import numpy as np

model = None


def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    # AZUREML_MODEL_DIR points at the registered model's folder.
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        # Fallback: search subfolders (registered models are nested by version)
        for root, _, files in os.walk(model_dir):
            if "model.pkl" in files:
                model_path = os.path.join(root, "model.pkl")
                break
    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)
        instances = np.array(data["data"])
        predictions = model.predict(instances)
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})
