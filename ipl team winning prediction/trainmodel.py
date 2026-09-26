import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("data/ipldata.csv")

# First 5 rows
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Shape
print("\nShape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns)



# Remove unnecessary columns
df.drop(columns=["mid", "date"], inplace=True)

print(df.head())
print(df.shape)


print(df.columns)

# Features
X = df.drop("total", axis=1)

# Target
y = df["total"]

print("X Shape:", X.shape)
print("y Shape:", y.shape)
# Check data types
print(X.dtypes)
categorical_cols = [
    "venue",
    "batting_team",
    "bowling_team",
    "batsman",
    "bowler"
]
numerical_cols = [
    "runs",
    "wickets",
    "overs",
    "runs_last_5",
    "wickets_last_5",
    "striker",
    "non-striker"
]
print("Categorical Columns:")
print(categorical_cols)

print("\nNumerical Columns:")
print(numerical_cols)
print(X.dtypes)

categorical_cols = [
    "venue",
    "batting_team",
    "bowling_team",
    "batsman",
    "bowler"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ],
    remainder="passthrough"
)

print("Preprocessor Created Successfully!")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
))
])

print("Model Pipeline Created Successfully!")
# print("Training Started...")

model.fit(X_train, y_train)

# print("Training Completed!")
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)

joblib.dump(model, "ipl_score_predictor.pkl")

print("Model Saved Successfully!")
import pandas as pd
import joblib

# Load model
model = joblib.load("ipl_score_predictor.pkl")

sample = pd.DataFrame({
    "venue": ["M Chinnaswamy Stadium"],
    "batting_team": ["Royal Challengers Bangalore"],
    "bowling_team": ["Chennai Super Kings"],
    "batsman": ["V Kohli"],
    "bowler": ["DJ Bravo"],
    "runs": [120],
    "wickets": [3],
    "overs": [15.0],
    "runs_last_5": [45],
    "wickets_last_5": [1],
    "striker": [65],
    "non-striker": [30]
})

prediction = model.predict(sample)

print("Predicted Final Score:", round(prediction[0]))