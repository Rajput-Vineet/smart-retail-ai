import pandas as pd


# =========================
# LOAD ASSOCIATION RULES
# =========================

rules = pd.read_csv(
    "data/processed/association_rules.csv"
)

print("=" * 50)
print("ASSOCIATION RULES LOADED")
print("=" * 50)

print(rules.shape)


# =========================
# CLEAN STRING COLUMNS
# =========================

rules["antecedents"] = (
    rules["antecedents"]
    .astype(str)
)

rules["consequents"] = (
    rules["consequents"]
    .astype(str)
)


# =========================
# FORMAT PRODUCT NAMES
# =========================

def clean_product_name(text):

    text = text.replace(
        "frozenset({",
        ""
    )

    text = text.replace(
        "})",
        ""
    )

    text = text.replace(
        "'",
        ""
    )

    return text


# =========================
# RECOMMENDATION FUNCTION
# =========================

def recommend_products(
    product_name,
    top_n=5
):

    recommendations = rules[
        rules["antecedents"]
        .str.contains(
            product_name,
            case=False,
            na=False
        )
    ]

    recommendations = recommendations.sort_values(
        by=["lift", "confidence"],
        ascending=False
    )

    print("\n" + "=" * 50)
    print(f"RECOMMENDATIONS FOR:")
    print(product_name)
    print("=" * 50)

    if recommendations.empty:
        print("\nNo recommendations found.")
        return

    recommendations = recommendations.head(top_n)

    for i, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        consequent = clean_product_name(
            row["consequents"]
        )

        print(f"\n{i}. Recommended Product(s):")
        print(f"   {consequent}")

        print(
            f"   Confidence: "
            f"{row['confidence']:.2f}"
        )

        print(
            f"   Lift Score: "
            f"{row['lift']:.2f}"
        )

        print(
            f"   Support: "
            f"{row['support']:.3f}"
        )


# =========================
# TEST RECOMMENDATIONS
# =========================

recommend_products(
    "WHITE HANGING HEART T-LIGHT HOLDER"
)

recommend_products(
    "JUMBO BAG RED RETROSPOT"
)

recommend_products(
    "PARTY BUNTING"
)