import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# =========================
# LOAD RFM DATA
# =========================

rfm = pd.read_csv(
    "data/processed/rfm_features.csv"
)

print("=" * 50)
print("ORIGINAL RFM DATA")
print("=" * 50)

print(rfm.head())


# =========================
# LOG TRANSFORMATION
# Handle skewed monetary/frequency
# =========================

rfm["Recency"] = np.log1p(rfm["Recency"])
rfm["Frequency"] = np.log1p(rfm["Frequency"])
rfm["Monetary"] = np.log1p(rfm["Monetary"])


print("\n" + "=" * 50)
print("AFTER LOG TRANSFORMATION")
print("=" * 50)

print(rfm.describe())


# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm[["Recency", "Frequency", "Monetary"]]
)

# Convert back to dataframe
rfm_scaled = pd.DataFrame(
    rfm_scaled,
    columns=["Recency", "Frequency", "Monetary"]
)

# Add CustomerID back
rfm_scaled["CustomerID"] = rfm["CustomerID"]


# =========================
# FINAL OUTPUT
# =========================

print("\n" + "=" * 50)
print("SCALED RFM FEATURES")
print("=" * 50)

print(rfm_scaled.head())


# =========================
# SAVE SCALER
# =========================

joblib.dump(
    scaler,
    "models/preprocessing/rfm_scaler.pkl"
)

print("\nScaler saved successfully!")


# =========================
# SAVE SCALED DATA
# =========================

rfm_scaled.to_csv(
    "data/processed/rfm_scaled.csv",
    index=False
)

print("\nScaled RFM data saved successfully!")