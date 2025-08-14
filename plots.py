import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import pandas as pd

def plot_matching_queries(matched_df: pd.DataFrame, save_plot: bool = True) -> None:
    """
    Create visualizations for matching queries showing performance regressions.

    Args:
        matched_df (pd.DataFrame): DataFrame with matched queries
        save_plot (bool): Whether to save the plot to file
    """
    if len(matched_df) == 0:
        print("No matching queries found with performance regressions.")
        return

    # Filter out any infinite or NaN values that might cause plotting issues
    plot_df = matched_df[
        (matched_df['performance_ratio'].notna()) &
        (matched_df['performance_ratio'] != float('inf')) &
        (matched_df['performance_ratio'] != float('-inf'))
    ].copy()

    if len(plot_df) == 0:
        print("No valid data for plotting after filtering infinite values.")
        return

    print(f"Plotting {len(plot_df)} valid regression entries...")

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Query Performance Regression Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Scatter plot comparing 1.0.6 vs 1.1.0 performance
    ax1.scatter(plot_df['ms_1_0_6'], plot_df['ms_1_1_0'],
               alpha=0.7, s=50, color='red')

    # Add diagonal line (y=x) for reference
    min_ms = min(plot_df['ms_1_0_6'].min(), plot_df['ms_1_1_0'].min())
    max_ms = max(plot_df['ms_1_0_6'].max(), plot_df['ms_1_1_0'].max())
    ax1.plot([min_ms, max_ms], [min_ms, max_ms], 'k--', alpha=0.5, label='Equal Performance')

    ax1.set_xlabel('1.0.6 Query Time (ms)')
    ax1.set_ylabel('1.1.0 Query Time (ms)')
    ax1.set_title('Query Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance regression distribution
    ax2.hist(plot_df['performance_regression'], bins=30, alpha=0.7,
             color='orange', edgecolor='black')
    ax2.set_xlabel('Performance Regression (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Performance Regressions')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Performance ratio distribution (cap extreme values for better visualization)
    ratio_data = plot_df['performance_ratio']
    # Cap ratios at 95th percentile for better visualization
    ratio_cap = ratio_data.quantile(0.95)
    ratio_capped = ratio_data.clip(upper=ratio_cap)

    ax3.hist(ratio_capped, bins=30, alpha=0.7,
             color='purple', edgecolor='black')
    ax3.set_xlabel(f'Performance Ratio (1.1.0 / 1.0.6) [capped at {ratio_cap:.1f}x]')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Performance Ratios')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Time series of regressions
    ax4.scatter(plot_df['timestamp_1_0_6'], plot_df['performance_regression'],
               alpha=0.7, s=30, color='red', label='Regression Amount')
    ax4.set_xlabel('Timestamp (1.0.6)')
    ax4.set_ylabel('Performance Regression (ms)')
    ax4.set_title('Performance Regressions Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Format x-axis for better readability
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if save_plot:
        plt.savefig('query_performance_regression.png', dpi=300, bbox_inches='tight')
        print("Regression analysis graph saved as 'query_performance_regression.png'")

    plt.show()



def create_timing_graphs(df1: pd.DataFrame, df2: pd.DataFrame,
                        file1_name: str = "1.0.6", file2_name: str = "1.1.0",
                        save_plots: bool = True) -> None:
    """
    Create graphs comparing timing data from two DataFrames.

    Args:
        df1 (pd.DataFrame): First DataFrame with timestamp and milliseconds
        df2 (pd.DataFrame): Second DataFrame with timestamp and milliseconds
        file1_name (str): Name for first dataset (default: "1.0.6")
        file2_name (str): Name for second dataset (default: "1.1.0")
        save_plots (bool): Whether to save plots to files (default: True)
    """
    # Filter out rows with missing data
    df1_valid = df1.dropna(subset=['timestamp', 'milliseconds'])
    df2_valid = df2.dropna(subset=['timestamp', 'milliseconds'])

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Log Performance Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Time series for both datasets
    if len(df1_valid) > 0:
        ax1.scatter(df1_valid['timestamp'], df1_valid['milliseconds'],
                   alpha=0.6, s=20, color='blue', label=file1_name)
    if len(df2_valid) > 0:
        ax1.scatter(df2_valid['timestamp'], df2_valid['milliseconds'],
                   alpha=0.6, s=20, color='red', label=file2_name)

    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Milliseconds')
    ax1.set_title('Query Performance Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Format x-axis for better readability
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Histogram comparison
    if len(df1_valid) > 0:
        ax2.hist(df1_valid['milliseconds'], bins=200, alpha=0.7,
                color='blue', label=file1_name, density=True)
    if len(df2_valid) > 0:
        ax2.hist(df2_valid['milliseconds'], bins=200, alpha=0.7,
                color='red', label=file2_name, density=True)

    ax2.set_xlabel('Milliseconds')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Query Times')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Individual time series for file 1
    if len(df1_valid) > 0:
        ax3.plot(df1_valid['timestamp'], df1_valid['milliseconds'],
                'o-', alpha=0.7, color='blue', markersize=3, linewidth=1)
        ax3.set_xlabel('Timestamp')
        ax3.set_ylabel('Milliseconds')
        ax3.set_title(f'{file1_name} Query Performance')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # Plot 4: Individual time series for file 2
    if len(df2_valid) > 0:
        ax4.plot(df2_valid['timestamp'], df2_valid['milliseconds'],
                'o-', alpha=0.7, color='red', markersize=3, linewidth=1)
        ax4.set_xlabel('Timestamp')
        ax4.set_ylabel('Milliseconds')
        ax4.set_title(f'{file2_name} Query Performance')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if save_plots:
        plt.savefig('log_performance_comparison.png', dpi=300, bbox_inches='tight')
        print("Graph saved as 'log_performance_comparison.png'")

    plt.show()

def create_summary_stats_plot(stats1: dict, stats2: dict,
                             file1_name: str = "1.0.6", file2_name: str = "1.1.0",
                             save_plot: bool = True) -> None:
    """
    Create a bar chart comparing summary statistics between two datasets.

    Args:
        stats1 (dict): Statistics dictionary for first dataset
        stats2 (dict): Statistics dictionary for second dataset
        file1_name (str): Name for first dataset
        file2_name (str): Name for second dataset
        save_plot (bool): Whether to save plot to file
    """
    # Extract metrics for comparison
    metrics = ['avg_ms', 'median_ms', 'min_ms', 'max_ms', 'std_ms']
    values1 = [stats1.get(metric, 0) for metric in metrics]
    values2 = [stats2.get(metric, 0) for metric in metrics]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(metrics))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], values1, width,
                   label=file1_name, color='blue', alpha=0.7)
    bars2 = ax.bar([i + width/2 for i in x], values2, width,
                   label=file2_name, color='red', alpha=0.7)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Milliseconds')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Average', 'Median', 'Minimum', 'Maximum', 'Std Dev'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)

    plt.tight_layout()

    if save_plot:
        plt.savefig('performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Statistics graph saved as 'performance_metrics_comparison.png'")

    plt.show()