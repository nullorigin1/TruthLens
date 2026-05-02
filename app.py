from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
with open("../models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        news = request.form["news"]
        vect = vectorizer.transform([news])
        pred = model.predict(vect)

        prediction = "Real News ✅" if pred[0] == 1 else "Fake News ❌"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)