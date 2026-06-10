from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -----------------------------
# Load model and encoders
# -----------------------------
model = load_model("mushroom_pca_model.h5")

with open("pca_model.pkl", "rb") as f:
    pca = pickle.load(f)

with open("ohe.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            request.form.get("cap_shape"),
            request.form.get("cap_surface"),
            request.form.get("cap_color"),
            request.form.get("bruises"),
            request.form.get("odor"),
            request.form.get("gill_attachment"),
            request.form.get("gill_spacing"),
            request.form.get("gill_size"),
            request.form.get("gill_color"),
            request.form.get("stalk_shape"),
            request.form.get("stalk_root"),
            request.form.get("stalk_surface_above_ring"),
            request.form.get("stalk_surface_below_ring"),
            request.form.get("stalk_color_above_ring"),
            request.form.get("stalk_color_below_ring"),
            request.form.get("veil_type"),
            request.form.get("veil_color"),
            request.form.get("ring_number"),
            request.form.get("ring_type"),
            request.form.get("spore_print_color"),
            request.form.get("population"),
            request.form.get("habitat")
        ]

        input_df = pd.DataFrame([features])

        encoded = ohe.transform(input_df)
        transformed = pca.transform(encoded)

        prediction = model.predict(transformed)
        probability = float(prediction[0][0])

        result = "☠️ Poisonous Mushroom" if probability >= 0.5 else "🍄 Edible Mushroom"

        return render_template("index.html",
                               prediction=result,
                               probability=round(probability, 4))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0")
