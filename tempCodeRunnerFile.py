from flask import Flask, render_template, request
import numpy as np
import joblib

# Load model
model = joblib.load('model/ipl_score_predictor_model.pkl')

# Flask app
app = Flask(__name__)

# Team encoding (must match training order and drop='first')
teams = [
    'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 'Kolkata Knight Riders',
    'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

@app.route('/')
def home():
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    runs = int(request.form['runs'])
    wickets = int(request.form['wickets'])
    overs = float(request.form['overs'])
    runs_last_5 = int(request.form['runs_last_5'])
    wickets_last_5 = int(request.form['wickets_last_5'])

    input_vector = []

    # One-hot encode batting and bowling team (drop first to match model)
    for team in teams[1:]:  # drop='first'
        input_vector.append(1 if batting_team == team else 0)
    for team in teams[1:]:
        input_vector.append(1 if bowling_team == team else 0)

    # Add numerical values
    input_vector += [runs, wickets, overs, runs_last_5, wickets_last_5]

    # Predict
    prediction = model.predict([input_vector])[0]
    predicted_score = int(round(prediction))

    return render_template('index.html', prediction=predicted_score, teams=teams)

if __name__ == '__main__':
    app.run(debug=True)
