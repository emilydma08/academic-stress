from flask import Flask, render_template, request
import torch
from training.model import NeuralNetwork
import joblib
import numpy as np
import os

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

@app.route("/form", methods=["GET"])
def form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_values = [
        float(request.form["study_hours"]),
        float(request.form["ec_hours"]),
        float(request.form["sleep_hours"]),
        float(request.form["social_hours"]),
        float(request.form["exercise_hours"]),
        float(request.form["gpa"]),
    ]

    input_array = np.array([input_values])  
    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    # Map prediction back to classification group
    class_map = {0: "Low", 1: "Moderate", 2: "High"}
    subtexts = ["Great job! Your lifestyle habits seem to be keeping your stress under control. Keep maintaining healthy routines around sleep, studying, and social time to stay balanced.", "You’re managing, but some habits may be pushing your stress higher than ideal. Try adjusting one area—like adding more breaks, exercise, or consistent sleep—to bring your stress down.", "Your results suggest that stress is taking a toll. It may help to step back, prioritize rest, and talk to someone you trust. Small changes to sleep, workload balance, or self-care can make a big difference."]
    predicted_label = class_map[predicted_class]

    return render_template("results.html", prediction=predicted_label, prediction_subtext=subtexts[predicted_class])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
