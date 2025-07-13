import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Load the dataset
df = pd.read_csv("ipl_colab.csv")

# Keep only necessary columns
df = df[['batting_team', 'bowling_team', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total']]

# Filter to valid consistent IPL teams
valid_teams = [
    'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 'Kolkata Knight Riders',
    'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]
df = df[df['batting_team'].isin(valid_teams) & df['bowling_team'].isin(valid_teams)]

# Split into features and label
X = df.drop('total', axis=1)
y = df['total']

# One-hot encode the team names (batting + bowling), drop first to prevent dummy trap
encoder = OneHotEncoder(categories=[valid_teams, valid_teams], drop='first', sparse_output=False)
teams_encoded = encoder.fit_transform(X[['batting_team', 'bowling_team']])

# Generate column names for encoded features
encoded_columns = (
    [f"batting_{team}" for team in valid_teams[1:]] +
    [f"bowling_{team}" for team in valid_teams[1:]]
)
encoded_df = pd.DataFrame(teams_encoded, columns=encoded_columns)

# Extract and reset numerical feature index
numerical = X[['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5']].reset_index(drop=True)

# Combine encoded and numerical features
X_final = pd.concat([encoded_df, numerical], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/ipl_score_predictor_model.pkl")

print(" Model trained and saved to 'model/ipl_score_predictor_model.pkl'")
