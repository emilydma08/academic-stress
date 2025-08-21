from flask import Flask, render_template, request
import torch
from training.model import NeuralNetwork
import joblib
import numpy as np

app = Flask(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load("best_model.pth", map_location=device, weights_only=False)

best_config = checkpoint['config']
state_dict = checkpoint['model_state_dict']

model = NeuralNetwork(
    input_size=6, 
    hidden_dim=best_config['num_hidden_units'],
    num_layers=best_config['num_hidden_layers'],
    dropout_rate=best_config['dropout']
).to(device)

model.load_state_dict(state_dict)
model.eval()

scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # grab form values
    input_values = [
        float(request.form["study_hours"]),
        float(request.form["ec_hours"]),
        float(request.form["sleep_hours"]),
        float(request.form["social_hours"]),
        float(request.form["exercise_hours"]),
        float(request.form["gpa"]),
    ]

    input_array = np.array([input_values])  # shape (1, 6)
    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    # Map numeric prediction back to label
    class_map = {0: "Low", 1: "Moderate", 2: "High"}
    predicted_label = class_map[predicted_class]

    return render_template("results.html", prediction=predicted_label)


if __name__ == "__main__":
    app.run(debug=True)
