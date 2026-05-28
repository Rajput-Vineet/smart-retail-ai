import joblib
import pandas as pd

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules


# =========================
# LOAD CLEANED DATA
# =========================

df = pd.read_csv(
    "data/processed/cleaned_retail_data.csv"
)

print("=" * 50)
print("DATASET LOADED")
print("=" * 50)

print(df.shape)


# =========================
# CREATE BASKET MATRIX
# =========================

basket = (
    df.groupby(["InvoiceNo", "Description"])["Quantity"]
    .sum()
    .unstack()
    .fillna(0)
)

# Convert quantities to binary
basket = basket > 0

print("\nBasket Shape:")
print(basket.shape)


# =========================
# APPLY FP-GROWTH
# =========================

frequent_items = fpgrowth(
    basket,
    min_support=0.02,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_items.head())


# =========================
# GENERATE ASSOCIATION RULES
# =========================

rules = association_rules(
    frequent_items,
    metric="lift",
    min_threshold=1
)

# Sort by lift
rules = rules.sort_values(
    by="lift",
    ascending=False
)

print("\n" + "=" * 50)
print("TOP ASSOCIATION RULES")
print("=" * 50)

print(
    rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ].head(10)
)


# =========================
# SAVE RULES OBJECT
# =========================

joblib.dump(
    rules,
    "models/recommendation/association_rules.pkl"
)

print("\nAssociation rules model saved!")


# =========================
# SAVE RULES
# =========================

rules.to_csv(
    "data/processed/association_rules.csv",
    index=False
)

print("\nAssociation rules saved successfully!")