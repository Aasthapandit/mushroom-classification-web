from flask import Flask, render_template, request

app = Flask(**name**)

@app.route("/")
def home():
return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
return render_template(
"index.html",
prediction="✅ Predict Route Working",
probability="100%"
)

if **name** == "**main**":
app.run(debug=True)
