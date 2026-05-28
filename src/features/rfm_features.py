import pandas as pd


# =========================
# LOAD CLEANED DATA
# =========================

df = pd.read_csv(
    "data/processed/cleaned_retail_data.csv"
)

# Convert date column again after loading CSV
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])


# =========================
# CREATE REFERENCE DATE
# (latest purchase date + 1 day)
# =========================

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

print("=" * 50)
print("REFERENCE DATE")
print("=" * 50)
print(reference_date)


# =========================
# CREATE RFM FEATURES
# =========================

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
})

# Rename columns
rfm.columns = ["Recency", "Frequency", "Monetary"]


# =========================
# BASIC INFO
# =========================

print("\n" + "=" * 50)
print("RFM DATASET")
print("=" * 50)

print(rfm.head())

print("\nShape:")
print(rfm.shape)

print("\nStatistics:")
print(rfm.describe())


# =========================
# SAVE RFM DATA
# =========================

rfm.to_csv(
    "data/processed/rfm_features.csv"
)

print("\nRFM features saved successfully!")