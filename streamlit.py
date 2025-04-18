# Install required packages
# pip install matplotlib seaborn streamlit pandas numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set the title of the Streamlit app
st.title("Aviation KPIs Dashboard")

# Cache the data loading function for performance
@st.cache_data
def load_data():
    # Load the dataset
    df = pd.read_csv("Aviation_KPIs_Dataset.xlsx - Sheet1.csv")
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert datetime columns if they exist
    if "Actual Departure Time" in df.columns:
        df["Actual Departure Time"] = pd.to_datetime(df["Actual Departure Time"], errors='coerce')
        df["Month"] = df["Actual Departure Time"].dt.month
    if "Scheduled Departure Time" in df.columns:
        df["Scheduled Departure Time"] = pd.to_datetime(df["Scheduled Departure Time"], errors='coerce')
        df["Scheduled Month"] = df["Scheduled Departure Time"].dt.month
        # Map months to seasons
        season_mapping = {
            1: "Winter", 2: "Winter", 3: "Spring",
            4: "Spring", 5: "Spring", 6: "Summer",
            7: "Summer", 8: "Summer", 9: "Autumn",
            10: "Autumn", 11: "Autumn", 12: "Winter"
        }
        df["Season"] = df["Scheduled Month"].map(season_mapping)
    return df

# Load the data
df = load_data()

# Sidebar: Let the user choose which plot to display
plot_option = st.sidebar.selectbox(
    "Select a plot to display",
    [
        "Maintenance Downtime vs. Profit", 
        "Revenue vs. Profit Distribution",
        "Distribution of Profit",
        "Feature Correlation Heatmap",
        "Load Factor vs Profit (Outliers Check)",
        "Revenue vs Operating Cost (Profitable vs Non-Profitable Flights)",
        "Monthly Profit Trends",
        "Impact of Flight Delays on Profit",
        "Fuel Efficiency vs. Profitability",
        "Seasonal Profit Analysis",
        "All"
    ]
)

# Function to display Maintenance Downtime vs. Profit
def plot_maintenance_downtime_vs_profit():
    st.header("Maintenance Downtime vs. Profit")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Maintenance Downtime (Hours)", y="Profit (USD)", data=df, ax=ax)
    ax.set_title("Maintenance Downtime vs. Profit")
    st.pyplot(fig)

# Function to display Revenue vs. Profit Distribution
def plot_revenue_vs_profit_distribution():
    st.header("Revenue vs. Profit Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=df["Revenue (USD)"],
        y=df["Profit (USD)"],
        hue=df["Profit (USD)"] > 0,
        palette={True: "green", False: "red"},
        ax=ax
    )
    ax.set_title("Revenue (USD) vs. Profit (USD) (Profitable vs. Non-Profitable Flights)")
    ax.set_xlabel("Revenue (USD)")
    ax.set_ylabel("Profit (USD)")
    st.pyplot(fig)

# Function to display Distribution of Profit
def plot_distribution_of_profit():
    st.header("Distribution of Profit")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Profit (USD)'], bins=30, kde=True, color='lightblue', ax=ax)
    ax.set_title("Distribution of Profit")
    ax.set_xlabel("Profit (USD)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Function to display Feature Correlation Heatmap
def plot_feature_correlation_heatmap():
    st.header("Feature Correlation Heatmap")
    corr_matrix = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

# Function to display Load Factor vs Profit (Outliers Check)
def plot_load_factor_vs_profit():
    st.header("Load Factor vs Profit (Outliers Check)")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=df["Load Factor (%)"], y=df["Profit (USD)"], ax=ax)
    ax.set_title("Load Factor vs Profit (Outliers Check)")
    ax.set_xlabel("Load Factor (%)")
    ax.set_ylabel("Profit (USD)")
    st.pyplot(fig)

# Function to display Revenue vs Operating Cost
def plot_revenue_vs_operating_cost():
    st.header("Revenue vs Operating Cost (Profitable vs Non-Profitable Flights)")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x=df["Revenue (USD)"],
        y=df["Operating Cost (USD)"],
        hue=df["Profit (USD)"] > 0,
        palette={True: "green", False: "red"},
        ax=ax
    )
    ax.set_title("Revenue (USD) vs Operating Cost (Profitable vs Non-Profitable Flights)")
    ax.set_xlabel("Revenue (USD)")
    ax.set_ylabel("Operating Cost (USD)")
    st.pyplot(fig)

# Function to display Monthly Profit Trends
def plot_monthly_profit_trends():
    if "Month" in df.columns:
        st.header("Monthly Profit Trends")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=df["Month"], y=df["Profit (USD)"], estimator=np.mean, ci=None, marker="o", ax=ax)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_title("Monthly Profit Trends")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Profit (USD)")
        st.pyplot(fig)
    else:
        st.write("The 'Month' column is not available in the dataset.")

# Function to display Impact of Flight Delays on Profit
def plot_impact_of_flight_delays():
    st.header("Impact of Flight Delays on Profit")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df["Delay (Minutes)"], y=df["Profit (USD)"], ax=ax)
    ax.set_title("Impact of Flight Delays on Profit")
    ax.set_xlabel("Flight Delay (Minutes)")
    ax.set_ylabel("Profit (USD)")
    st.pyplot(fig)

# Function to display Fuel Efficiency vs. Profitability
def plot
::contentReference[oaicite:0]{index=0}
 
