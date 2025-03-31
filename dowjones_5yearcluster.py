import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
import pandas as pd

# Function to calculate expected annualized return with projections
def calculate_annualized_return_with_projection(ticker, start_date, end_date, projection_end_date):
    # Download historical price data
    data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']

    if data.empty:
        print(f"No data for {ticker}")
        return np.nan

    # Prepare data for Prophet
    df = data.reset_index()
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})

    # Fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Create future dataframe with projections
    future = model.make_future_dataframe(periods=(pd.to_datetime(projection_end_date) - pd.to_datetime(end_date)).days)
    forecast = model.predict(future)

    # Extract the projected data
    projected_data = forecast.set_index('ds')['yhat']

    # Combine historical and projected data
    combined_data = pd.concat([data, projected_data[projection_end_date:]], axis=0)

    # Calculate daily returns
    daily_returns = combined_data.pct_change().dropna()

    # Calculate the average daily return
    average_daily_return = daily_returns.mean()

    # Calculate the expected annualized return
    trading_days = 252  # Assuming 252 trading days in a year
    annualized_return = (1 + average_daily_return) ** trading_days - 1

    return annualized_return

# Parameters
start_date = "2020-01-01"  # Start date 5 years ago
end_date = "2025-03-31"  # End date one day before the current date
projection_end_date = "2030-12-31"  # End of projection period

# List of Dow Jones tickers organized in clusters
clusters = {
    "Cluster A": ["MSFT", "PG", "DIS", "UNH", "VZ"],
    "Cluster B": ["JPM", "HD", "KO", "MCD", "BA"],
    "Cluster C": ["NKE", "INTC", "CSCO", "WMT", "UNH"],
    "Cluster D": ["PFE", "XOM", "GS", "CVX", "IBM"],
    "Cluster E": ["CAT", "TRV", "AMGN", "UNP", "DOW"],
    "Cluster F": ["RTX", "WBA", "CVS", "C", "AAPL"]
}

# Initialize lists to store annualized returns and cluster information
all_annualized_returns = []
all_tickers = []
colors = []
cluster_labels = []

# Assign colors to clusters
cluster_colors = {
    "Cluster A": '#1f77b4',
    "Cluster B": '#ff7f0e',
    "Cluster C": '#2ca02c',
    "Cluster D": '#d62728',
    "Cluster E": '#9467bd',
    "Cluster F": '#8c564b'
}

# Calculate and store annualized returns for each ticker
for cluster_name, tickers in clusters.items():
    for ticker in tickers:
        annualized_return = calculate_annualized_return_with_projection(ticker, start_date, end_date,
                                                                        projection_end_date)
        if not np.isnan(annualized_return):  # Check if the return is valid
            all_annualized_returns.append(annualized_return * 100)  # Convert to percentage
            all_tickers.append(ticker)
            colors.append(cluster_colors[cluster_name])
            cluster_labels.append(cluster_name)

# Plotting the results
x = np.arange(len(all_tickers))  # the label locations

plt.figure(figsize=(24, 12))  # Increased figure width for better visibility

# Plotting each cluster with different colors and markers
for cluster_name in clusters.keys():
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_name]
    cluster_returns = [all_annualized_returns[i] for i in cluster_indices]
    cluster_x = np.array(cluster_indices)

    # Scatter plot
    plt.scatter(cluster_x, cluster_returns,
                color=cluster_colors[cluster_name], label=cluster_name, s=100, edgecolor='k', alpha=0.7)

    # Line plot connecting the dots within each cluster
    plt.plot(cluster_x, cluster_returns, color=cluster_colors[cluster_name], linestyle='-', linewidth=2)

# Adding a red dashed line at y = 20
plt.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Threshold Line (20%)')

# Adding title and labels
plt.title("Expected Annualized Returns for Dow Jones Stocks (2020-2030)", fontsize=18)
plt.xlabel("Company Ticker", fontsize=16)
plt.ylabel("Annualized Return (%)", fontsize=16)

# Set y-axis limits and ticks
min_value = -25  # Minimum limit for y-axis
max_value = max(all_annualized_returns)
y_padding = 10  # Additional space for better visibility
plt.ylim(min_value, max_value + y_padding)  # Adjust y-axis limits to include negative values

# Set y-axis ticks to go by 10% increments
plt.yticks(np.arange(min_value, max_value + y_padding + 10, 10))

# X-axis labels
plt.xticks(x, all_tickers, rotation=90, fontsize=10)

# Adding cluster labels as legend
plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

# Adding grid
plt.grid(True)

# Adjust layout to fit everything properly
plt.subplots_adjust(bottom=0.25, right=0.85)  # Adjust margins for better fitting of labels and legend

# Show plot
plt.show()
