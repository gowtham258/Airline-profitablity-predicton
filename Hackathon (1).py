# {
#  "cells": [
#   {
#    "cell_type": "code",
#    "execution_count": null,
#    "id": "6e698c40-de41-4e16-8742-919663d52923",
#    "metadata": {},
#    "outputs": [],
#    "source": []
#   }
#  ],
#  "metadata": {
#   "kernelspec": {
#    "display_name": "",
#    "name": ""
#   },
#   "language_info": {
#    "name": ""
#   }
#  },
#  "nbformat": 4,
#  "nbformat_minor": 5
# }

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime as dt
st.title("Aviation KPIs Dashboard")
df=pd.read_csv('Aviation_KPIs_Dataset.xlsx - Sheet1.csv')

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data loading (replace with your actual data loading mechanism)
# df = pd.read_csv("your_data.csv")
# For demonstration, we'll create a dummy DataFrame
data = {
    "Maintenance Downtime (Hours)": [5, 10, 3, 8, 6],
    "Profit (USD)": [20000, 15000, 25000, 18000, 22000]
}
df = pd.DataFrame(data)

# Title of the app
st.title("Maintenance Downtime vs. Profit")

# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="Maintenance Downtime (Hours)", y="Profit (USD)", data=df, ax=ax)
ax.set_title("Maintenance Downtime vs. Profit")

# Display the plot in the Streamlit app
st.pyplot(fig)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
# df = pd.read_csv("your_data.csv")  # Adjust file path and separator if needed

# Clean column names (remove extra spaces)
# df.columns = df.columns.str.strip()

# # Debug: Print column names to confirm
# st.write("Columns in the dataset:")
# st.write(df.columns.tolist())

# ---------------------------
# 1. Maintenance Downtime vs. Operating Cost
# ---------------------------
# st.header("Maintenance Downtime vs. Operating Cost")
# fig1, ax1 = plt.subplots(figsize=(8, 6))
# sns.scatterplot(x='Maintenance Downtime (Hours)', y='Operating Cost (USD)', data=df, ax=ax1)
# ax1.set_title("Maintenance Downtime vs. Operating Cost")
# st.pyplot(fig1)

# ---------------------------
# 2. Histogram of Delays
# ---------------------------
# st.header("Distribution of Delays")
# fig2, ax2 = plt.subplots(figsize=(8, 6))
# sns.histplot(df['Delay (Minutes)'], bins=20, kde=True, ax=ax2)
# ax2.set_title("Distribution of Delays")
# st.pyplot(fig2)

# ---------------------------
# 3. Operating Cost vs. Profit
# ---------------------------
# st.header("Operating Cost vs. Profit")
# fig3, ax3 = plt.subplots(figsize=(8, 6))
# sns.scatterplot(x='Operating Cost (USD)', y='Profit (USD)', data=df, ax=ax3)
# ax3.set_title("Operating Cost vs. Profit")
# ax3.set_xlabel("Operating Cost (USD)")
# ax3.set_ylabel("Profit (USD)")
# ax3.grid(True)
# st.pyplot(fig3)

# ---------------------------
# 4. Fleet Availability vs. Profit
# ---------------------------
# st.header("Fleet Availability vs. Profit")
# fig4, ax4 = plt.subplots(figsize=(8, 6))
# sns.lineplot(x='Fleet Availability (%)', y='Profit (USD)', data=df, ax=ax4)
# ax4.set_title("Fleet Availability vs. Profit")
# st.pyplot(fig4)


# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load your data - adjust the file path as needed
# # df = pd.read_csv("your_data.csv")  # Replace with your actual file path

# # Set the title for your Streamlit app
# st.title("Revenue vs. Profit Distribution")

# # Create a Matplotlib figure and axis
# fig, ax = plt.subplots(figsize=(10, 6))

# # Create the scatter plot with Seaborn
# sns.scatterplot(
#     x=df["Revenue (USD)"],
#     y=df["Profit (USD)"],
#     hue=df["Profit (USD)"] > 0,    # This will color points green for profitable flights and red for non-profitable
#     palette={True: "green", False: "red"},
#     ax=ax
# )

# # Set the title and axis labels
# ax.set_title("Revenue (USD) vs. Profit (USD) (Profitable vs. Non-Profitable Flights)")
# ax.set_xlabel("Revenue (USD)")
# ax.set_ylabel("Profit (USD)")

# # Display the plot in the Streamlit app
# st.pyplot(fig)


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Use Streamlit caching to speed up data loading
@st.cache_data
def load_data():
    # Adjust the file path as needed. Ensure the CSV file is in your working directory.
    df = pd.read_csv('Aviation_KPIs_Dataset.xlsx - Sheet1.csv')
    # Clean column names by stripping extra spaces
    df.columns = df.columns.str.strip()
    return df

# Load the data
df = load_data()

# Set the title of the dashboard
st.title("Aviation KPIs Dashboard")

# Sidebar: Let the user choose which plot to display, including a "Select All" option
plot_option = st.sidebar.selectbox(
    "Select a plot to display",
    [
        "Maintenance Downtime vs. Profit", 
        "Revenue vs. Profit Distribution",
        "Select All"
    ]
)

# Display the Maintenance Downtime vs. Profit plot if selected or if "Select All" is chosen
if plot_option == "Maintenance Downtime vs. Profit" or plot_option == "Select All":
    st.header("Maintenance Downtime vs. Profit")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Maintenance Downtime (Hours)", y="Profit (USD)", data=df, ax=ax)
    ax.set_title("Maintenance Downtime vs. Profit")
    st.pyplot(fig)

# Display the Revenue vs. Profit Distribution plot if selected or if "Select All" is chosen
if plot_option == "Revenue vs. Profit Distribution" or plot_option == "Select All":
    st.header("Revenue vs. Profit Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=df["Revenue (USD)"],
        y=df["Profit (USD)"],
        hue=df["Profit (USD)"] > 0,  # Color points based on profit status
        palette={True: "green", False: "red"},
        ax=ax
    )
    ax.set_title("Revenue (USD) vs. Profit (USD) (Profitable vs. Non-Profitable Flights)")
    ax.set_xlabel("Revenue (USD)")
    ax.set_ylabel("Profit (USD)")
    st.pyplot(fig)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data - update the file path as necessary
df = pd.read_csv("Aviation_KPIs_Dataset.xlsx - Sheet1.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Set the title for the Streamlit app
st.title("Distribution of Profit")

# Create the histogram plot
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['Profit (USD)'], bins=30, kde=True, color='lightblue', ax=ax)
ax.set_title("Distribution of Profit")
ax.set_xlabel("Profit (USD)")
ax.set_ylabel("Frequency")

# Display the plot in the Streamlit app
st.pyplot(fig)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data - adjust the file path as needed
df = pd.read_csv("Aviation_KPIs_Dataset.xlsx - Sheet1.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Set the title for the Streamlit app
st.title("Feature Correlation Heatmap")

# Compute the correlation matrix for numeric columns only
corr_matrix = df.corr(numeric_only=True)

# Create a Matplotlib figure and axis for the heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap")

# Display the heatmap in the Streamlit app
st.pyplot(fig)
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cache the data loading for performance
@st.cache_data
def load_data():
    # Update the file path as needed
    df = pd.read_csv("Aviation_KPIs_Dataset.xlsx - Sheet1.csv")
    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()
    
    # Convert 'Actual Departure Time' to datetime if present, then extract Month
    if "Actual Departure Time" in df.columns:
        df["Actual Departure Time"] = pd.to_datetime(df["Actual Departure Time"])
        df["Month"] = df["Actual Departure Time"].dt.month
    return df

# Load the data
# df = load_data()

# Set the dashboard title
st.title("Aviation KPIs Visualizations")

# -----------------------------------
# 1. Boxplot for Outlier Detection (Load Factor vs. Profit)
# -----------------------------------
st.header("Load Factor vs Profit (Outliers Check)")
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.boxplot(x=df["Load Factor (%)"], y=df["Profit (USD)"], ax=ax1)
ax1.set_title("Load Factor vs Profit (Outliers Check)")
ax1.set_xlabel("Load Factor (%)")
ax1.set_ylabel("Profit (USD)")
st.pyplot(fig1)

# -----------------------------------
# 2. Revenue vs Operating Cost (Scatterplot)
# -----------------------------------
st.header("Revenue vs Operating Cost (Profitable vs Non-Profitable Flights)")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    x=df["Revenue (USD)"],
    y=df["Operating Cost (USD)"],
    hue=df["Profit (USD)"] > 0,  # Points will be green if profit is positive, red otherwise
    palette={True: "green", False: "red"},
    ax=ax2
)
ax2.set_title("Revenue (USD) vs Operating Cost (Profitable vs Non-Profitable Flights)")
ax2.set_xlabel("Revenue (USD)")
ax2.set_ylabel("Operating Cost (USD)")
st.pyplot(fig2)

# -----------------------------------

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cache the data loading for better performance
@st.cache_data
def load_data():
    # Adjust the file path as needed
    df = pd.read_csv("Aviation_KPIs_Dataset.xlsx - Sheet1.csv")
    # Clean column names by stripping extra spaces
    df.columns = df.columns.str.strip()
    
    # Convert "Actual Departure Time" to datetime and extract month
    if "Actual Departure Time" in df.columns:
        df["Actual Departure Time"] = pd.to_datetime(df["Actual Departure Time"])
        df["Month"] = df["Actual Departure Time"].dt.month
    return df

# Load the data
df = load_data()

# Set the title for the Streamlit app
st.title("Monthly Profit Trends Dashboard")

# Check if the "Month" column exists and then plot
if "Month" in df.columns:
    st.header("Monthly Profit Trends")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create a line plot for average profit by month
    sns.lineplot(x=df["Month"], y=df["Profit (USD)"], estimator=np.mean, ci=None, marker="o", ax=ax)
    
    # Customize the x-axis to show month names
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title("Monthly Profit Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Profit (USD)")
    
    # Render the plot in Streamlit
    st.pyplot(fig)
else:
    st.write("The 'Month' column is not available in the dataset.")

# -----------------------------------
# 4. Impact of Flight Delays on Profit (Boxplot)
# -----------------------------------
st.header("Impact of Flight Delays on Profit")
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.boxplot(x=df["Delay (Minutes)"], y=df["Profit (USD)"], ax=ax4)
ax4.set_title("Impact of Flight Delays on Profit")
ax4.set_xlabel("Flight Delay (Minutes)")
ax4.set_ylabel("Profit (USD)")
st.pyplot(fig4)

# -----------------------------------
# 5. Fuel Efficiency vs. Profitability (Scatterplot)
# -----------------------------------
st.header("Fuel Efficiency vs. Profitability")
fig5, ax5 = plt.subplots(figsize=(12, 6))
sns.scatterplot(
    x=df["Fuel Efficiency (ASK)"],
    y=df["Profit (USD)"],
    hue=df["Profit (USD)"] > 0,
    palette={True: "green", False: "red"},
    ax=ax5
)
ax5.set_title("Fuel Efficiency vs. Profitability")
ax5.set_xlabel("Fuel Efficiency (Miles per Gallon)")
ax5.set_ylabel("Profit (USD)")
st.pyplot(fig5)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cache the data loading for improved performance
@st.cache_data
def load_data():
    # Adjust the file path as needed
    df = pd.read_csv("Aviation_KPIs_Dataset.xlsx - Sheet1.csv")
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert Scheduled Departure Time to datetime
    df["Scheduled Departure Time"] = pd.to_datetime(df["Scheduled Departure Time"])
    
    # Extract the month from Scheduled Departure Time
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

# Load the dataset
df = load_data()

# Set the dashboard title
st.title("Seasonal Profit Analysis")

# Add a button for showing the plot
if st.button("Show Total Profit per Season"):
    # Create the bar plot for total profit per season
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=df["Season"], y=df["Profit (USD)"], estimator=sum, ci=None, palette="coolwarm", ax=ax)
    ax.set_title("Total Profit per Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Total Profit (USD)")
    
    # Display the plot in the Streamlit app
    st.pyplot(fig)
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cache data loading for performance
@st.cache_data
def load_data():
    # Adjust the file path as needed
    df = pd.read_csv("Aviation_KPIs_Dataset.xlsx - Sheet1.csv")
    # Clean column names by stripping extra spaces
    df.columns = df.columns.str.strip()
    
    # Convert Scheduled Departure Time to datetime
    df["Scheduled Departure Time"] = pd.to_datetime(df["Scheduled Departure Time"])
    
    # Extract the month from Scheduled Departure Time
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

# Load the dataset
df = load_data()

# Set the title for the dashboard
st.title("Profit Distribution by Season")

# Add a button to the sidebar to show the plot
if st.sidebar.button("Show Profit Distribution by Season"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df["Season"], y=df["Profit (USD)"], palette="Set2", ax=ax)
    ax.set_title("Profit Distribution Across Seasons")
    ax.set_xlabel("Season")
    ax.set_ylabel("Profit (USD)")
    st.pyplot(fig)


