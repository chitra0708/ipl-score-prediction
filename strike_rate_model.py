import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

dataset_path = "datasets/mensIPLHawkeyeStats.csv"

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")
print("Total records:", len(df))


# ==========================================
# 2. CHECK COLUMNS
# ==========================================

if "batterRuns" not in df.columns:
    raise ValueError("batterRuns column was not found.")

if "batter" not in df.columns:
    raise ValueError("batter column was not found.")


# ==========================================
# 3. SELECT DATA
# ==========================================

df = df[["batter", "batterRuns"]].copy()

df["batterRuns"] = pd.to_numeric(
    df["batterRuns"],
    errors="coerce"
)

df = df.dropna()

print("Records after cleaning:", len(df))


# ==========================================
# 4. BALL NUMBER
# ==========================================

df["BallNumber"] = (
    df.groupby("batter").cumcount() + 1
)


# ==========================================
# 5. CUMULATIVE RUNS
# ==========================================

df["CumulativeRuns"] = (
    df.groupby("batter")["batterRuns"].cumsum()
)


# ==========================================
# 6. STRIKE RATE
# ==========================================

df["StrikeRate"] = (
    df["CumulativeRuns"] /
    df["BallNumber"]
) * 100


# ==========================================
# 7. REMOVE INVALID VALUES
# ==========================================

df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

df = df.dropna()


# ==========================================
# 8. FEATURES
# ==========================================

X = df[
    [
        "BallNumber",
        "CumulativeRuns"
    ]
]

y = df["StrikeRate"]


# ==========================================
# 9. TRAIN / TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 10. MODEL
# ==========================================

model = LinearRegression()


# ==========================================
# 11. TRAIN
# ==========================================

print()
print("Training Strike Rate model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================
# 12. TEST
# ==========================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


# ==========================================
# 13. PERFORMANCE
# ==========================================

print()
print("====================================")
print("STRIKE RATE MODEL PERFORMANCE")
print("====================================")

print(
    "Mean Absolute Error:",
    round(mae, 2)
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
    "strike_rate_model.pkl"
)

print()
print("====================================")
print("SUCCESS!")
print("====================================")

print("strike_rate_model.pkl created successfully!")

print("====================================")