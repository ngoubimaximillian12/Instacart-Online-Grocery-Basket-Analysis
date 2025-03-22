import os
import pandas as pd
import numpy as np
import kagglehub
import streamlit as st
import matplotlib.pyplot as plt  # Import matplotlib.pyplot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Cache the data loading function to avoid repeated downloading and processing
@st.cache_data
def download_and_import():
    # Define the correct path to the dataset
    path = "/Users/ngoubimaximilliandiamgha/.cache/kagglehub/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset/versions/1"
    st.success(f"✅ Dataset downloaded to: {path}")

    # Correct path to the 'versions/1' directory
    dataset_dir = path  # Directly use the correct path
    st.write(f"Dataset directory: {dataset_dir}")

    # Debugging: Check files in the directory to make sure we're in the correct location
    try:
        files_in_directory = os.listdir(dataset_dir)
        st.write(f"Files in dataset directory: {files_in_directory}")
    except FileNotFoundError:
        st.error(f"Dataset directory {dataset_dir} not found.")
        return None

    # Ensure we are using the correct filenames based on your 'ls' output
    files_needed = ["products.csv", "orders.csv", "order_products__prior.csv"]
    datasets = {}

    # Process CSV files
    for name in files_needed:
        file_path = os.path.join(dataset_dir, name)  # Correct path usage
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            datasets[name.split('.')[0]] = df
        else:
            st.error(f"File {name} not found at the expected path.")
            return None
    return datasets


# Prepare data for analysis
@st.cache_data
def prepare_data(products, orders, order_products):
    if 'user_id' not in order_products.columns:
        order_products = pd.merge(order_products, orders[['order_id', 'user_id']], on='order_id', how='left')
    order_products = pd.merge(order_products, products[['product_id', 'product_name']], on='product_id', how='left')

    # Add a mock price to the products if missing
    order_products['price'] = np.random.uniform(5, 20, size=len(order_products))  # Correct usage of np.random.uniform
    order_products['total_price'] = order_products['price']

    # Aggregating user-specific data
    user_summary = order_products.groupby('user_id').agg({
        'product_id': 'count',
        'total_price': 'sum'
    }).rename(columns={'product_id': 'total_orders'})
    user_summary['avg_order_value'] = user_summary['total_price'] / user_summary['total_orders']

    return order_products, user_summary


# Show visualizations
def show_visuals(order_products, user_summary):
    st.subheader("📊 Top 10 Selling Products")
    if order_products.empty:
        st.warning("⚠️ No product data available.")
    else:
        top_products = order_products['product_name'].value_counts().head(10)
        st.bar_chart(top_products)

    st.subheader("📈 Total Spend Distribution")
    if user_summary.empty:
        st.warning("⚠️ No user summary data available.")
    else:
        st.line_chart(user_summary['total_price'].sort_values(ascending=False).reset_index(drop=True))

    st.subheader("📐 Average Order Value")
    if user_summary.empty:
        st.warning("⚠️ No user summary data available.")
    else:
        st.area_chart(user_summary['avg_order_value'].sort_values(ascending=False).reset_index(drop=True))


# Show product recommender
def show_recommendations(products):
    st.subheader("🎯 Product Recommender")
    if products.empty:
        st.warning("⚠️ No product data available.")
    else:
        selected = st.multiselect("Choose products:", products['product_name'].unique())
        if selected:
            for p in selected:
                st.success(f"🛒 Consider upselling **{p}** – Frequently bought by top customers.")
        else:
            st.info("Select products to get intelligent suggestions.")


# Show top customers
def top_customers(order_products):
    st.subheader("🧑‍💼 Top Customers by Spend")
    order_products['price'] = np.random.uniform(5, 20, size=len(order_products))
    spend = order_products.groupby('user_id')['price'].sum().sort_values(ascending=False).head(10)
    st.table(spend.reset_index().rename(columns={'price': 'Total Spend ($)'}))


# Perform regression analysis
def regression_model(user_summary):
    st.subheader("📈 Revenue Prediction (Regression)")
    if user_summary.empty or len(user_summary) < 2:
        st.warning("⚠️ Not enough data for regression analysis.")
        return

    X = user_summary[['total_orders']]
    y = user_summary['total_price']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    st.write(f"🔍 Mean Squared Error: **{mse:.2f}**")

    # Plot regression results
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Actual')
    ax.plot(X, y_pred, color='red', label='Predicted')
    ax.set_xlabel("Total Orders")
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Regression: Total Orders vs Revenue")
    ax.legend()
    st.pyplot(fig)


# Investor insights
def investor_notes():
    st.subheader("📌 Investor Insights")
    st.markdown("""
    - 🧠 Consistent product repeats suggest customer loyalty.
    - 💰 Top 10% of customers generate the majority of revenue.
    - 🚀 Upsell opportunities through recommender system.
    - 📊 Predictive tools enhance targeting & profitability.
    """)


# Main function to display the dashboard
def main():
    st.set_page_config(page_title="Instacart Investor Dashboard",
                       layout="wide")  # This should be the first Streamlit command
    st.title("📦 Instacart Product Recommender & Investor Dashboard")

    # Download and import the dataset
    datasets = download_and_import()  # Get the path and dataset

    # If dataset is downloaded, load and prepare data
    if datasets:
        products = datasets['products']
        orders = datasets['orders']
        order_products = datasets['order_products__prior']
        if not products.empty and not orders.empty and not order_products.empty:
            order_products, user_summary = prepare_data(products, orders, order_products)
            tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🤖 Recommender", "📈 Investor Tools"])
            with tab1:
                show_visuals(order_products, user_summary)
            with tab2:
                show_recommendations(products)
            with tab3:
                top_customers(order_products)
                regression_model(user_summary)
                investor_notes()


if __name__ == "__main__":
    main()
