import pandas as pd
import re
from datetime import datetime
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

def extract_timestamp_and_ms(csv_file_path: str) -> pd.DataFrame:
    """
    Extract timestamp and milliseconds from the textPayload column of a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: DataFrame with columns 'timestamp', 'milliseconds', and 'original_text'
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Check if textPayload column exists
    if 'textPayload' not in df.columns:
        raise ValueError("textPayload column not found in CSV file")

    # Initialize lists to store extracted data
    timestamps = []
    milliseconds = []
    original_texts = []

    # Regular expression pattern to match timestamp and milliseconds
    # Pattern: '2025-08-13 11:59:58.946+0000 INFO  0 ms:
    pattern = r"'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\+\d{4}) INFO\s+(\d+) ms:"

    for idx, text_payload in enumerate(df['textPayload']):
        if pd.isna(text_payload):
            timestamps.append(None)
            milliseconds.append(None)
            original_texts.append(None)
            continue

        # Extract timestamp and milliseconds using regex
        match = re.search(pattern, str(text_payload))

        if match:
            timestamp_str = match.group(1)
            ms_value = int(match.group(2))

            # Parse timestamp to datetime object
            try:
                # Remove timezone offset for parsing
                timestamp_clean = timestamp_str.replace('+0000', '')
                timestamp_dt = datetime.strptime(timestamp_clean, '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(timestamp_dt)
                milliseconds.append(ms_value)
                original_texts.append(text_payload)
            except ValueError as e:
                print(f"Error parsing timestamp at row {idx}: {e}")
                timestamps.append(None)
                milliseconds.append(None)
                original_texts.append(text_payload)
        else:
            # No match found
            timestamps.append(None)
            milliseconds.append(None)
            original_texts.append(text_payload)

    # Create result DataFrame
    result_df = pd.DataFrame({
        'timestamp': timestamps,
        'milliseconds': milliseconds,
        'original_text': original_texts
    })

    return result_df

def parse_log_files_comparison(file1_path: str, file2_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse two log CSV files and return DataFrames with extracted timestamp and milliseconds.

    Args:
        file1_path (str): Path to first CSV file
        file2_path (str): Path to second CSV file

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of DataFrames for file1 and file2
    """
    df1 = extract_timestamp_and_ms(file1_path)
    df2 = extract_timestamp_and_ms(file2_path)

    return df1, df2

def get_performance_stats(df: pd.DataFrame) -> dict:
    """
    Get basic performance statistics from the extracted data.

    Args:
        df (pd.DataFrame): DataFrame with timestamp and milliseconds columns

    Returns:
        dict: Dictionary with performance statistics
    """
    valid_rows = df.dropna(subset=['milliseconds'])

    if len(valid_rows) == 0:
        return {"error": "No valid millisecond data found"}

    stats = {
        "total_entries": len(df),
        "valid_entries": len(valid_rows),
        "avg_ms": valid_rows['milliseconds'].mean(),
        "median_ms": valid_rows['milliseconds'].median(),
        "min_ms": valid_rows['milliseconds'].min(),
        "max_ms": valid_rows['milliseconds'].max(),
        "std_ms": valid_rows['milliseconds'].std()
    }

    return stats

def extract_query_text(text_payload: str) -> str:
    """
    Extract the query part from textPayload (everything after "> neo4j - ").

    Args:
        text_payload (str): The full textPayload string

    Returns:
        str: The extracted query text, or empty string if not found
    """
    if pd.isna(text_payload):
        return ""

    # Find the pattern "> neo4j - " and extract everything after it
    pattern = r'>\s*neo4j\s+- (.+)'
    match = re.search(pattern, str(text_payload))
    # print(f"Extracting query from textPayload: {text_payload[100:300]}...")  # Debug output

    if match:
        return match.group(1).strip()
    return ""

def find_matching_queries(df1: pd.DataFrame, df2: pd.DataFrame,
                         time_threshold_ms: int = 500) -> pd.DataFrame:
    """
    Find queries that appear in both datasets within a time window and where
    1.1.0 query time is greater than 1.0.6 query time.

    Args:
        df1 (pd.DataFrame): DataFrame for 1.0.6 logs
        df2 (pd.DataFrame): DataFrame for 1.1.0 logs
        time_threshold_ms (int): Time window in milliseconds for matching queries

    Returns:
        pd.DataFrame: DataFrame with matched queries and their performance comparison
    """
    # Filter valid entries
    df1_valid = df1.dropna(subset=['timestamp', 'milliseconds', 'original_text']).copy()
    df2_valid = df2.dropna(subset=['timestamp', 'milliseconds', 'original_text']).copy()

    # Extract query text for both datasets
    df1_valid['query_text'] = df1_valid['original_text'].apply(extract_query_text)
    df2_valid['query_text'] = df2_valid['original_text'].apply(extract_query_text)

    # Remove entries with empty query text or zero milliseconds (to avoid inf ratios)
    df1_valid = df1_valid[(df1_valid['query_text'] != "") & (df1_valid['milliseconds'] > 0)]
    df2_valid = df2_valid[(df2_valid['query_text'] != "") & (df2_valid['milliseconds'] > 0)]

    matched_queries = []
    used_df2_indices = set()  # Track which df2 entries we've already matched

    print(f"Searching for matches in {len(df1_valid)} df1 entries and {len(df2_valid)} df2 entries...")

    for idx1, row1 in df1_valid.iterrows():
        query1 = row1['query_text']
        timestamp1 = row1['timestamp']
        ms1 = row1['milliseconds']

        best_match = None
        best_time_diff = float('inf')

        # Find the best matching query in df2 (closest in time)
        for idx2, row2 in df2_valid.iterrows():
            if idx2 in used_df2_indices:
                continue  # Skip already matched entries

            query2 = row2['query_text']
            timestamp2 = row2['timestamp']
            ms2 = row2['milliseconds']

            # Check if queries match exactly
            if query1 == query2:
                time_diff = abs((timestamp2 - timestamp1).total_seconds() * 1000)

                # Only consider if within time threshold and 1.1.0 is slower
                if time_diff <= time_threshold_ms and ms2 > ms1:
                    if time_diff < best_time_diff:
                        best_match = {
                            'idx2': idx2,
                            'query_text': query1,
                            'timestamp_1_0_6': timestamp1,
                            'timestamp_1_1_0': timestamp2,
                            'ms_1_0_6': ms1,
                            'ms_1_1_0': ms2,
                            'time_diff_ms': time_diff,
                            'performance_regression': ms2 - ms1,
                            'performance_ratio': ms2 / ms1
                        }
                        best_time_diff = time_diff

        # Add the best match if found
        if best_match:
            used_df2_indices.add(best_match['idx2'])
            del best_match['idx2']  # Remove internal tracking field
            matched_queries.append(best_match)

    print(f"Found {len(matched_queries)} unique matching queries with performance regressions")
    return pd.DataFrame(matched_queries)

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

def analyze_query_regressions(matched_df: pd.DataFrame) -> dict:
    """
    Analyze the query performance regressions and return summary statistics.

    Args:
        matched_df (pd.DataFrame): DataFrame with matched queries

    Returns:
        dict: Dictionary with regression analysis statistics
    """
    if len(matched_df) == 0:
        return {"error": "No matched queries found"}

    # Filter out infinite values for meaningful statistics
    valid_ratios = matched_df[
        (matched_df['performance_ratio'].notna()) &
        (matched_df['performance_ratio'] != float('inf')) &
        (matched_df['performance_ratio'] != float('-inf'))
    ]['performance_ratio']

    analysis = {
        "total_matched_queries": len(matched_df),
        "avg_regression_ms": matched_df['performance_regression'].mean(),
        "median_regression_ms": matched_df['performance_regression'].median(),
        "max_regression_ms": matched_df['performance_regression'].max(),
        "min_regression_ms": matched_df['performance_regression'].min(),
        "avg_performance_ratio": valid_ratios.mean() if len(valid_ratios) > 0 else 0,
        "median_performance_ratio": valid_ratios.median() if len(valid_ratios) > 0 else 0,
        "max_performance_ratio": valid_ratios.max() if len(valid_ratios) > 0 else 0,
        "worst_regression_query": matched_df.loc[matched_df['performance_regression'].idxmax(), 'query_text'][:100] + "...",
        "queries_with_2x_slowdown": len(matched_df[matched_df['performance_ratio'] >= 2.0]),
        "queries_with_5x_slowdown": len(matched_df[matched_df['performance_ratio'] >= 5.0]),
        "queries_with_10x_slowdown": len(matched_df[matched_df['performance_ratio'] >= 10.0])
    }

    return analysis

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

# Example usage
if __name__ == "__main__":
    # Example of how to use the function
    try:
        print("Extracting data from CSV files...")

        # Extract data from both files
        df1 = extract_timestamp_and_ms("1.0.6-logs.csv")
        df2 = extract_timestamp_and_ms("1.1.0-logs.csv")

        print(f"Extracted {len(df1)} entries from 1.0.6-logs.csv")
        print(f"Valid entries with timestamp/ms: {df1.dropna(subset=['milliseconds']).shape[0]}")

        print(f"Extracted {len(df2)} entries from 1.1.0-logs.csv")
        print(f"Valid entries with timestamp/ms: {df2.dropna(subset=['milliseconds']).shape[0]}")

        # Get performance stats
        stats1 = get_performance_stats(df1)
        stats2 = get_performance_stats(df2)

        print("\n1.0.6 Performance Stats:")
        for key, value in stats1.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        print("\n1.1.0 Performance Stats:")
        for key, value in stats2.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        # Show sample of extracted data
        print("\nSample extracted data from 1.0.6:")
        sample_df1 = df1[['timestamp', 'milliseconds']].dropna().head()
        print(sample_df1)

        print("\nSample extracted data from 1.1.0:")
        sample_df2 = df2[['timestamp', 'milliseconds']].dropna().head()
        print(sample_df2)

        # Find matching queries with performance regressions
        print("\nFinding matching queries with performance regressions...")
        matched_queries = find_matching_queries(df1, df2, time_threshold_ms=500)

        if len(matched_queries) > 0:
            print(f"Found {len(matched_queries)} matching queries where 1.1.0 is slower than 1.0.6")

            # Analyze regressions
            regression_analysis = analyze_query_regressions(matched_queries)
            print("\nRegression Analysis:")
            for key, value in regression_analysis.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

            # Show top 5 worst regressions
            print("\nTop 5 Worst Performance Regressions:")
            top_regressions = matched_queries.nlargest(5, 'performance_regression')
            for idx, row in top_regressions.iterrows():
                print(f"  Regression: {row['performance_regression']:.0f}ms "
                      f"({row['ms_1_0_6']:.0f}ms -> {row['ms_1_1_0']:.0f}ms, "
                      f"ratio: {row['performance_ratio']:.2f}x)")
                print(f"    Query: {row['query_text'][:80]}...")
                print()

            # Create regression analysis plots
            print("\nGenerating regression analysis graphs...")
            plot_matching_queries(matched_queries)
        else:
            print("No matching queries found with performance regressions.")

        # Create graphs
        print("\nGenerating performance comparison graphs...")
        create_timing_graphs(df1, df2, "1.0.6", "1.1.0")

        print("\nGenerating summary statistics comparison...")
        create_summary_stats_plot(stats1, stats2, "1.0.6", "1.1.0")

        print("\nAnalysis complete! Check the generated PNG files for visual comparisons.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
