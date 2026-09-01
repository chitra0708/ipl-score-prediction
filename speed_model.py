import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

dataset_path = "datasets/mensIPLHawkeyeStats.csv"

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")
print("Total records:", len(df))


# ==========================================
# 2. SELECT USEFUL COLUMNS
# ==========================================

features = [
    "bowler",
    "bowlingStyle",
    "rightArmedBowl",
    "ball",
    "pitchX",
    "pitchY"
]

target = "ballSpeed"


# ==========================================
# 3. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = features + [target]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            "Column not found: " + column
        )


# ==========================================
# 4. CLEAN DATA
# ==========================================

df = df[required_columns].copy()

df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)

df["pitchX"] = pd.to_numeric(
    df["pitchX"],
    errors="coerce"
)

df["pitchY"] = pd.to_numeric(
    df["pitchY"],
    errors="coerce"
)

df["ball"] = pd.to_numeric(
    df["ball"],
    errors="coerce"
)


# Remove invalid speed values

df = df[
    (df[target] > 0) &
    (df[target] < 50)
]


# Remove missing values

df = df.dropna()


print("Records after cleaning:", len(df))


# ==========================================
# 5. INPUT AND TARGET
# ==========================================

X = df[features]

y = df[target]


# ==========================================
# 6. CATEGORICAL FEATURES
# ==========================================

categorical_features = [
    "bowler",
    "bowlingStyle"
]


# ==========================================
# 7. NUMERICAL FEATURES
# ==========================================

numerical_features = [
    "rightArmedBowl",
    "ball",
    "pitchX",
    "pitchY"
]


# ==========================================
# 8. PREPROCESSING
# ==========================================

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


# ==========================================
# 9. RANDOM FOREST MODEL
# ==========================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ==========================================
# 10. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 11. TRAIN MODEL
# ==========================================

print("\nTraining Ball Speed model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================
# 12. TEST MODEL
# ==========================================

y_pred = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


# ==========================================
# 13. MODEL PERFORMANCE
# ==========================================

print("\n====================================")
print("BALL SPEED MODEL PERFORMANCE")
print("====================================")

print(
    "Mean Absolute Error:",
    round(mae, 3),
    "m/s"
)

print(
    "R2 Score:",
    round(r2, 3)
)


# ==========================================
# 14. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "speed_model.pkl"
)


print("\n====================================")
print("SUCCESS!")
print("====================================")

print("speed_model.pkl created successfully!")

print("====================================")