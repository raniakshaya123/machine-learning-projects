from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load("ipl_score_predictor.pkl")

# Load Dataset
df = pd.read_csv("data/ipldata.csv")

@app.route('/')
def home():
    return render_template(
        "index.html",
        venues=sorted(df["venue"].unique()),
        batting_teams=sorted(df["batting_team"].unique()),
        bowling_teams=sorted(df["bowling_team"].unique()),
        batsmen=sorted(df["batsman"].unique()),
        bowlers=sorted(df["bowler"].unique())
    )

@app.route('/predict', methods=['POST'])
def predict():

    venue = request.form["venue"]
    batting_team = request.form["batting_team"]
    bowling_team = request.form["bowling_team"]
    batsman = request.form["batsman"]
    bowler = request.form["bowler"]

    runs = int(request.form["runs"])
    wickets = int(request.form["wickets"])
    overs = float(request.form["overs"])
    runs_last_5 = int(request.form["runs_last_5"])
    wickets_last_5 = int(request.form["wickets_last_5"])
    striker = int(request.form["striker"])
    non_striker = int(request.form["non_striker"])

    sample = pd.DataFrame({
        "venue": [venue],
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "batsman": [batsman],
        "bowler": [bowler],
        "runs": [runs],
        "wickets": [wickets],
        "overs": [overs],
        "runs_last_5": [runs_last_5],
        "wickets_last_5": [wickets_last_5],
        "striker": [striker],
        "non-striker": [non_striker]
    })

    prediction = round(model.predict(sample)[0])

    return render_template(
        "index.html",
        prediction=prediction,
        venues=sorted(df["venue"].unique()),
        batting_teams=sorted(df["batting_team"].unique()),
        bowling_teams=sorted(df["bowling_team"].unique()),
        batsmen=sorted(df["batsman"].unique()),
        bowlers=sorted(df["bowler"].unique())
    )

if __name__ == "__main__":
    app.run(debug=True)