import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("datasets/IPL_Score_Prediction_Dataset.csv")


# ==============================
# 2. SELECT REQUIRED COLUMNS
# ==============================

features = [
    "BattingTeam",
    "BowlingTeam",
    "CurrentScore",
    "Overs",
    "Wickets",
    "RunRate"
]

target = "FinalScore"

df = df[features + [target]]

# Remove incomplete rows
df = df.dropna()


# ==============================
# 3. INPUT AND OUTPUT
# ==============================

X = df[features]
y = df[target]


# ==============================
# 4. CATEGORICAL + NUMERICAL
# ==============================

categorical_features = [
    "BattingTeam",
    "BowlingTeam"
]

numerical_features = [
    "CurrentScore",
    "Overs",
    "Wickets",
    "RunRate"
]


# ==============================
# 5. PREPROCESSING
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "teams",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==============================
# 6. LINEAR REGRESSION MODEL
# ==============================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ==============================
# 7. TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# 8. TRAIN MODEL
# ==============================

model.fit(X_train, y_train)


# ==============================
# 9. TEST MODEL
# ==============================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Training Completed!")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# ==============================
# 10. SAVE MODEL
# ==============================

joblib.dump(model, "ipl_model.pkl")

print("ipl_model.pkl created successfully!")