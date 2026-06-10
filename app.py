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

    features = [
        request.form["cap_shape"],
        request.form["cap_surface"],
        request.form["cap_color"],
        request.form["bruises"],
        request.form["odor"],
        request.form["gill_attachment"],
        request.form["gill_spacing"],
        request.form["gill_size"],
        request.form["gill_color"],
        request.form["stalk_shape"],
        request.form["stalk_root"],
        request.form["stalk_surface_above_ring"],
        request.form["stalk_surface_below_ring"],
        request.form["stalk_color_above_ring"],
        request.form["stalk_color_below_ring"],
        request.form["veil_type"],
        request.form["veil_color"],
        request.form["ring_number"],
        request.form["ring_type"],
        request.form["spore_print_color"],
        request.form["population"],
        request.form["habitat"]
    ]

    columns = [
        'cap-shape',
        'cap-surface',
        'cap-color',
        'bruises',
        'odor',
        'gill-attachment',
        'gill-spacing',
        'gill-size',
        'gill-color',
        'stalk-shape',
        'stalk-root',
        'stalk-surface-above-ring',
        'stalk-surface-below-ring',
        'stalk-color-above-ring',
        'stalk-color-below-ring',
        'veil-type',
        'veil-color',
        'ring-number',
        'ring-type',
        'spore-print-color',
        'population',
        'habitat'
    ]

    input_df = pd.DataFrame([features], columns=columns)

    encoded = ohe.transform(input_df)
    transformed = pca.transform(encoded)

    prediction = model.predict(transformed)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        result = "☠️ Poisonous Mushroom"
    else:
        result = "🍄 Edible Mushroom"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability, 4)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0")
