import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


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
# CREATE CUSTOMER-PRODUCT MATRIX
# =========================

customer_product_matrix = (
    df.groupby(
        ["CustomerID", "Description"]
    )["Quantity"]
    .sum()
    .unstack()
    .fillna(0)
)

print("\nCustomer-Product Matrix Shape:")
print(customer_product_matrix.shape)


# =========================
# COMPUTE COSINE SIMILARITY
# =========================

customer_similarity = cosine_similarity(
    customer_product_matrix
)

similarity_df = pd.DataFrame(
    customer_similarity,
    index=customer_product_matrix.index,
    columns=customer_product_matrix.index
)

print("\nCustomer Similarity Matrix Created!")


# =========================
# RECOMMEND PRODUCTS
# =========================

def recommend_for_customer(
    customer_id,
    top_n=5
):

    if customer_id not in similarity_df.index:

        print("\nCustomer not found.")
        return

    # Get similar customers
    similar_customers = (
    similarity_df[customer_id]
    .sort_values(ascending=False)
    .iloc[1:6]
)

    print("\n" + "=" * 50)
    print(f"SIMILAR CUSTOMERS FOR {customer_id}")
    print("=" * 50)

    print(similar_customers)

    # Products bought by target customer
    customer_products = set(
        customer_product_matrix.loc[
            customer_id
        ][
            customer_product_matrix.loc[
                customer_id
            ] > 0
        ].index
    )

    recommendations = {}

    # Find products from similar customers
    for similar_customer in similar_customers.index:

        similar_customer_products = set(
            customer_product_matrix.loc[
                similar_customer
            ][
                customer_product_matrix.loc[
                    similar_customer
                ] > 0
            ].index
        )

        # Recommend unseen products
        unseen_products = (
            similar_customer_products
            - customer_products
        )

        for product in unseen_products:

            if product not in recommendations:
                recommendations[product] = 0

            recommendations[product] += 1

    # Sort recommendations
    recommendations = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 50)
    print("RECOMMENDED PRODUCTS")
    print("=" * 50)

    for product, score in recommendations[:top_n]:

        print(f"\n{product}")
        print(f"Recommendation Score: {score}")



# =========================
# TEST RECOMMENDATIONS
# =========================

recommend_for_customer(12747.0)



# =========================
# SAVE ARTIFACTS
# =========================

joblib.dump(
    similarity_df,
    "models/recommendation/customer_similarity.pkl"
)

joblib.dump(
    customer_product_matrix,
    "models/recommendation/customer_product_matrix.pkl"
)

print("\nCollaborative filtering artifacts saved!")