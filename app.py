from flask import Flask, render_template, request

app = Flask(**name**)

# Home Page

@app.route("/")
def home():
return render_template("index.html")

# Test Predict Route

@app.route("/predict", methods=["POST"])
def predict():
return render_template(
"index.html",
prediction="✅ Predict Route Working",
probability="100%"
)

# Run App

if **name** == "**main**":
app.run(debug=True)
