import os

from flasgger import Swagger, swag_from
from flask import Flask, jsonify, request

from src.models import ModelSelector
from src.utils import data_to_features, read_config_file

app = Flask(__name__)
swagger = Swagger(app)

# For demo, we will use the same configuration file and a fixed model save directory
config_path = "src/configs/config.yaml"
model_save_dir = "demo_space/saved_models/20241024_080540"

# Load configuration
config = read_config_file(config_path)
categorical_features = config["feature_engineering"]["categorical_features"]

# Read model type from model_save_dir
with open(os.path.join(model_save_dir, "model_type.txt"), "r") as f:
    model_type = f.read().strip()

model_selector = ModelSelector(model_type=model_type)
model_loader = model_selector.get_model()
model = model_loader.load_model(os.path.join(model_save_dir, "model.json"))


@app.route("/predict", methods=["POST"])
@swag_from("./swagger_config.yml")  # Reference the external swagger configuration
def predict():
    """
    Endpoint to make predictions using the model.
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        # Converts the dictionary to a pandas DataFrame and then to a numpy array
        x = data_to_features(data=data, feature_order=categorical_features)
        predictions = model.predict(x)

        # Return predictions as JSON
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
