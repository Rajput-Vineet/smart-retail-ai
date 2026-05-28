import streamlit as st
import pandas as pd
import plotly.express as px
import joblib


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="SmartRetail AI",
    layout="wide",
    page_icon="🛒"
)


# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():

    customers = pd.read_csv(
        "data/processed/customer_segments.csv"
    )

    retail = pd.read_csv(
        "data/processed/cleaned_retail_data.csv"
    )

    retail["InvoiceDate"] = pd.to_datetime(
        retail["InvoiceDate"]
    )

    return customers, retail


customers, retail = load_data()


# =========================
# SIDEBAR
# =========================

st.sidebar.title("🛒 SmartRetail AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Customer Segments",
        "Recommendations",
        "Customer AI"
    ]
)


# =========================
# EXECUTIVE DASHBOARD
# =========================

if page == "Executive Dashboard":

    st.title("📊 Executive Dashboard")

    total_revenue = retail["TotalPrice"].sum()

    total_customers = (
        retail["CustomerID"]
        .nunique()
    )

    total_transactions = (
        retail["InvoiceNo"]
        .nunique()
    )

    avg_order_value = (
        total_revenue
        / total_transactions
    )

    # KPI CARDS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        f"£{total_revenue:,.0f}"
    )

    col2.metric(
        "Customers",
        f"{total_customers:,}"
    )

    col3.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    col4.metric(
        "Avg Order Value",
        f"£{avg_order_value:.2f}"
    )

    st.markdown("---")

    # =========================
    # SEGMENT DISTRIBUTION
    # =========================

    st.subheader("Customer Segment Distribution")

    segment_counts = (
        customers["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Segment",
        "Count"
    ]

    fig = px.pie(
        segment_counts,
        names="Segment",
        values="Count",
        hole=0.4,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # =========================
    # PCA VISUALIZATION
    # =========================

    st.subheader("Customer Segmentation Map")

    fig2 = px.scatter(
        customers,
        x="PCA1",
        y="PCA2",
        color="Segment",
        hover_data=["CustomerID"],
        title="Customer Segments",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )

    # =========================
    # MONTHLY REVENUE ANALYSIS
    # =========================

    st.subheader("📈 Monthly Revenue Trend")

    retail["Month"] = (
        retail["InvoiceDate"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_revenue = (
        retail.groupby("Month")["TotalPrice"]
        .sum()
        .reset_index()
    )

    fig_monthly = px.line(
        monthly_revenue,
        x="Month",
        y="TotalPrice",
        markers=True,
        title="Monthly Revenue Over Time",
        template="plotly_dark"
    )

    fig_monthly.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue (£)"
    )

    st.plotly_chart(
        fig_monthly,
        width="stretch"
    )

    # =========================
    # TOP PRODUCTS
    # =========================

    st.subheader("🔥 Top Selling Products")

    top_products = (
        retail.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_products = px.bar(
        top_products,
        x="Quantity",
        y="Description",
        orientation="h",
        title="Top 10 Best Selling Products",
        template="plotly_dark"
    )

    fig_products.update_layout(
        yaxis=dict(
            autorange="reversed"
        )
    )

    st.plotly_chart(
        fig_products,
        width="stretch"
    )

    # =========================
    # REVENUE BY SEGMENT
    # =========================

    st.subheader("💰 Revenue by Customer Segment")

    segment_revenue = (
        customers.groupby("Segment")["Monetary"]
        .mean()
        .reset_index()
    )

    fig_segment = px.bar(
        segment_revenue,
        x="Segment",
        y="Monetary",
        color="Segment",
        title="Average Monetary Value by Segment",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_segment,
        width="stretch"
    )


# =========================
# CUSTOMER SEGMENTS PAGE
# =========================

elif page == "Customer Segments":

    st.title("👥 Customer Intelligence")

    st.dataframe(
        customers.head(20),
        width="stretch"
    )

    st.subheader("Segment Counts")

    st.bar_chart(
        customers["Segment"]
        .value_counts()
    )


# =========================
# RECOMMENDATIONS PAGE
# =========================

elif page == "Recommendations":

    st.title("🎯 Product Recommendation Engine")

    rules = joblib.load(
        "models/recommendation/association_rules.pkl"
    )

    rules["antecedents"] = (
        rules["antecedents"]
        .astype(str)
    )

    rules["consequents"] = (
        rules["consequents"]
        .astype(str)
    )

    product_input = st.text_input(
        "Enter Product Name"
    )

    if product_input:

        recommendations = rules[
            rules["antecedents"]
            .str.contains(
                product_input,
                case=False,
                na=False
            )
        ]

        recommendations = (
            recommendations
            .sort_values(
                by="lift",
                ascending=False
            )
            .head(5)
        )

        if recommendations.empty:

            st.warning(
                "No recommendations found."
            )

        else:

            st.subheader(
                "Recommended Products"
            )

            for _, row in recommendations.iterrows():

                st.markdown("---")

                # CLEAN PRODUCT NAME
                product_name = (
                    row["consequents"]
                    .replace("frozenset({'", "")
                    .replace("'})", "")
                )

                st.markdown(
                    f"### 🛍️ {product_name}"
                )

                st.write(
                    f"Confidence: "
                    f"{row['confidence']:.2f}"
                )

                st.write(
                    f"Lift Score: "
                    f"{row['lift']:.2f}"
                )


# =========================
# CUSTOMER AI PAGE
# =========================

elif page == "Customer AI":

    st.title("🧠 Customer Intelligence AI")

    # LOAD ARTIFACTS
    similarity_df = joblib.load(
        "models/recommendation/customer_similarity.pkl"
    )

    customer_product_matrix = joblib.load(
        "models/recommendation/customer_product_matrix.pkl"
    )

    # CUSTOMER INPUT
    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=12346.0,
        step=1.0
    )

    customer_id = float(customer_id)

    # CHECK CUSTOMER EXISTS
    if customer_id in customers["CustomerID"].values:

        customer_data = customers[
            customers["CustomerID"]
            == customer_id
        ].iloc[0]

        st.markdown("---")

        # KPI CARDS
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Segment",
            customer_data["Segment"]
        )

        col2.metric(
            "Recency",
            round(customer_data["Recency"], 2)
        )

        col3.metric(
            "Frequency",
            round(customer_data["Frequency"], 2)
        )

        col4.metric(
            "Monetary",
            round(customer_data["Monetary"], 2)
        )

        st.markdown("---")

        # =========================
        # SIMILAR CUSTOMERS
        # =========================

        st.subheader("👥 Similar Customers")

        similar_customers = (
            similarity_df[customer_id]
            .sort_values(ascending=False)
            .iloc[1:6]
        )

        similar_df = pd.DataFrame({
            "CustomerID":
                similar_customers.index,
            "Similarity Score":
                similar_customers.values
        })

        st.dataframe(
            similar_df,
            width="stretch"
        )

        # =========================
        # PRODUCT RECOMMENDATIONS
        # =========================

        st.subheader("🎯 Personalized Recommendations")

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

        for similar_customer in similar_customers.index:

            similar_products = set(
                customer_product_matrix.loc[
                    similar_customer
                ][
                    customer_product_matrix.loc[
                        similar_customer
                    ] > 0
                ].index
            )

            unseen_products = (
                similar_products
                - customer_products
            )

            for product in unseen_products:

                if product not in recommendations:
                    recommendations[product] = 0

                recommendations[product] += 1

        recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if recommendations:

            for product, score in recommendations[:5]:

                st.markdown("---")

                st.markdown(
                    f"### 🛍️ {product}"
                )

                st.write(
                    f"Recommendation Score: {score}"
                )

        else:

            st.warning(
                "No recommendations found."
            )

    else:

        st.error(
            "Customer ID not found."
        )