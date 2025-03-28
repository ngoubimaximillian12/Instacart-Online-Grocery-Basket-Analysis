# 🛍️ Instacart Investor Intelligence Dashboard

## Overview
The Instacart Investor Intelligence Dashboard is a Streamlit-powered interactive application designed to provide insightful data visualizations and actionable business intelligence based on the Instacart grocery dataset. It integrates data from Kaggle, PostgreSQL, and leverages machine learning models to offer valuable insights for investors.

## Key Features

- **Automated Data Acquisition & Storage:**
  - Downloads Instacart dataset directly from Kaggle using KaggleHub.
  - Imports and manages data storage seamlessly in PostgreSQL.

- **Comprehensive Data Insights:**
  - Monthly revenue trends visualization.
  - Top performing product departments.
  - Hourly order distribution analysis.
  - Regression analysis of order counts versus average spending.
  - Identification of top customers by total spend.
  - User order count distribution.

- **Investor-Focused Insights:**
  - Highlights key investment opportunities and strategic recommendations.
  - Provides actionable intelligence for increasing revenue through upselling and cross-selling strategies.

## Installation

Install the required Python packages:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn sqlalchemy psycopg2-binary kagglehub
```

## Usage

Run the Streamlit application:

```bash
streamlit run your_script.py
```

- Ensure PostgreSQL is installed and running with appropriate credentials.
- The app automatically handles data download, database setup, and visualizations.

## Requirements

- Python >= 3.7
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SQLAlchemy
- PostgreSQL
- KaggleHub

## Dataset

The dataset used is the Instacart Online Grocery Basket Analysis dataset, accessible directly from KaggleHub.

## File Structure

- `your_script.py`: Main Streamlit application script.
- PostgreSQL database to store and manage Instacart dataset.

## Author
Developed by Ngoubi Maximilliana Diangha .

## License
MIT License

