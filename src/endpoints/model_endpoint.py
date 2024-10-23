import os

import pandas as pd
from flask import Flask, jsonify, request

from src.data_handler import FeatureEngineer
from src.models import ModelSelector
from src.utils import read_config_file

app = Flask(__name__)

# For demo, we will use the same configuration file and model save directory
config_path = "src/configs/config.yaml"
model_save_dir = "demo_space/save_models/"

# Load configuration
config = read_config_file(config_path)

# Initialize components
feature_engineer = FeatureEngineer(config=config)

# read model _type from model_save_dir
with open(os.path.join(model_save_dir, "model_type.txt"), "r") as f:
    model_type = f.read().strip()

model_selector = ModelSelector(model_type=model_type)
model_loader = model_selector.get_model()
model = model_loader.load_model(os.path.join(model_save_dir, "model.json"))


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to make predictions using the model.
    :return: JSON with predictions or error message
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        # Converts the dictionary to a pandas DataFrame and then to a numpy array
        x = pd.DataFrame(data).values
        predictions = model.predict(x)

        # Return predictions as JSON
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
