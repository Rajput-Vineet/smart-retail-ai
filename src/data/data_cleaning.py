import pandas as pd


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/raw/OnlineRetail.csv",
    encoding="ISO-8859-1"
)

print("=" * 50)
print("INITIAL DATASET SHAPE")
print("=" * 50)
print(df.shape)


# =========================
# REMOVE MISSING CUSTOMER IDs
# =========================

df = df.dropna(subset=["CustomerID"])

print("\nAfter removing missing CustomerIDs:")
print(df.shape)


# =========================
# REMOVE MISSING DESCRIPTIONS
# =========================

df = df.dropna(subset=["Description"])

print("\nAfter removing missing descriptions:")
print(df.shape)


# =========================
# REMOVE CANCELLATIONS
# InvoiceNo starting with 'C'
# =========================

df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

print("\nAfter removing cancellations:")
print(df.shape)


# =========================
# REMOVE NEGATIVE OR ZERO QUANTITIES
# =========================

df = df[df["Quantity"] > 0]

print("\nAfter removing invalid quantities:")
print(df.shape)


# =========================
# REMOVE ZERO OR NEGATIVE PRICES
# =========================

df = df[df["UnitPrice"] > 0]

print("\nAfter removing invalid prices:")
print(df.shape)


# =========================
# KEEP ONLY UK TRANSACTIONS
# =========================

df = df[df["Country"] == "United Kingdom"]

print("\nAfter filtering UK transactions:")
print(df.shape)


# =========================
# CONVERT DATE COLUMN
# =========================

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    format="%m/%d/%y %H:%M"
)


# =========================
# CREATE TOTAL PRICE FEATURE
# =========================

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]


# =========================
# FINAL DATASET INFO
# =========================

print("\n" + "=" * 50)
print("FINAL CLEANED DATASET")
print("=" * 50)

print(df.head())

print("\nFinal Shape:")
print(df.shape)


# =========================
# SAVE CLEANED DATA
# =========================

df.to_csv(
    "data/processed/cleaned_retail_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")