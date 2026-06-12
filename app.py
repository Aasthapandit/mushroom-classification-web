from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -----------------------------
# LOAD MODELS
# -----------------------------
model = load_model("mushroom_pca_model.h5")

with open("pca_model.pkl", "rb") as f:
    pca = pickle.load(f)

with open("ohe.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


# -----------------------------
# HOME ROUTE
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# PREDICT ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
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
            'cap-shape','cap-surface','cap-color','bruises','odor',
            'gill-attachment','gill-spacing','gill-size','gill-color',
            'stalk-shape','stalk-root','stalk-surface-above-ring',
            'stalk-surface-below-ring','stalk-color-above-ring',
            'stalk-color-below-ring','veil-type','veil-color',
            'ring-number','ring-type','spore-print-color',
            'population','habitat'
        ]

        input_df = pd.DataFrame([features], columns=columns)

        encoded = ohe.transform(input_df).toarray()
        transformed = pca.transform(encoded)
        transformed = np.asarray(transformed, dtype=np.float32)

       # TEMP TEST
        prediction = [[0.95]]

        probability = float(prediction[0][0]) * 100

        if probability >= 50:
            result = "☠️ Poisonous Mushroom"
        else:
            result = "🍄 Edible Mushroom"

        return render_template(
            "index.html",
            prediction=result,
            probability=f"{probability:.2f}%"
        )

    except Exception as e:
        return f"ERROR: {str(e)}"


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
