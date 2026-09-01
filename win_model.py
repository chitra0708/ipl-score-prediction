import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv(
    "datasets/IPL_Score_Prediction_Dataset.csv"
)


# =========================================================
# 2. REQUIRED COLUMNS
# =========================================================

features = [
    "BattingTeam",
    "BowlingTeam",
    "CurrentScore",
    "Overs",
    "Wickets",
    "RunRate",
    "PitchType"
]


# =========================================================
# 3. CREATE WINNING TEAM
# =========================================================

# Dataset lo FinalScore based on batting team performance.
# Higher expected score -> batting team.
# Lower score -> bowling team.

df["WinningTeam"] = df.apply(
    lambda row:
        row["BattingTeam"]
        if row["FinalScore"] >= row["CurrentScore"]
        else row["BowlingTeam"],
    axis=1
)


# =========================================================
# 4. REMOVE INCOMPLETE DATA
# =========================================================

df = df.dropna(
    subset=features + ["WinningTeam"]
)


X = df[features]

y = df["WinningTeam"]


# =========================================================
# 5. CATEGORICAL FEATURES
# =========================================================

categorical_features = [
    "BattingTeam",
    "BowlingTeam",
    "PitchType"
]


numerical_features = [
    "CurrentScore",
    "Overs",
    "Wickets",
    "RunRate"
]


# =========================================================
# 6. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        )

    ],

    remainder="passthrough"
)


# =========================================================
# 7. LOGISTIC REGRESSION
# =========================================================

model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",

            LogisticRegression(
                max_iter=3000
            )
        )

    ]
)


# =========================================================
# 8. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


# =========================================================
# 9. TRAIN MODEL
# =========================================================

model.fit(
    X_train,
    y_train
)


# =========================================================
# 10. TEST MODEL
# =========================================================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "Winning Prediction Model Training Completed!"
)

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# =========================================================
# 11. SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "win_model.pkl"
)


print(
    "win_model.pkl created successfully!"
)