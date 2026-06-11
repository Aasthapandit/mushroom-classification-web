from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# ✅ FIRST CREATE APP
app = Flask(__name__)

# THEN LOAD MODEL
model = load_model("mushroom_pca_model.h5")

with open("pca_model.pkl", "rb") as f:
    pca = pickle.load(f)

with open("ohe.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


#@app.route("/predict", methods=["POST"])
def predict():
    return "Predict route reached"
# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
