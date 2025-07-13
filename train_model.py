import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Step 1: Load dataset
df = pd.read_csv("ipl_colab.csv")
print(" Dataset loaded. Shape:", df.shape)

# Step 2: Keep required columns only
df = df[['batting_team', 'bowling_team', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total']]
df.dropna(inplace=True)

# Step 3: Encode teams (OneHotEncoder with drop='first' for consistency)
teams = sorted(df['batting_team'].unique())
encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
encoded_teams = encoder.fit_transform(df[['batting_team', 'bowling_team']]).toarray()

# Step 4: Prepare input features
numerical_data = df[['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5']].values
X = pd.DataFrame(encoded_teams)
X = pd.concat([X, pd.DataFrame(numerical_data)], axis=1)
y = df['total']

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(" Model training complete.")

# Step 7: Save model with compression
os.makedirs("model", exist_ok=True)
joblib.dump(model, 'model/ipl_score_predictor_model.pkl', compress=9)
print(" Model saved to model/ipl_score_predictor_model.pkl")

# (Optional) Save encoder if needed for web input encoding
joblib.dump(encoder, 'model/team_encoder.pkl')
print(" Encoder saved to model/team_encoder.pkl")
