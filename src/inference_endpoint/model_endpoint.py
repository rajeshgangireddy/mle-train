import os
from flask import Flask, request, jsonify
import yaml
import pandas as pd
from src.data_handler import FeatureEngineer
from src.models import ModelSelector

app = Flask(__name__)

config_path="src/configs/config.yaml"
model_save_dir = "demo_space/save_models/"

# Load configuration
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Initialize components
feature_engineer = FeatureEngineer(config=config)

# read model _type from model_save_dir
with open(os.path.join(model_save_dir, "model_type.txt"), "r") as f:
    model_type = f.read().strip()

model_selector = ModelSelector(model_type=model_type)
model_loader = model_selector.get_model()
model = model_loader.load_model(os.path.join(model_save_dir, "model.json"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Make the data into an array -
        # this is required because the model expects a 2D array
        x = pd.DataFrame(data).values
        predictions = model.predict(x)

        # Return predictions as JSON
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)