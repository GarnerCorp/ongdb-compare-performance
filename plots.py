from matplotlib.pylab import rint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import pandas as pd

def plot_matching_queries(matched_df: pd.DataFrame, version1: str = "Version-1", version2: str = "Version-2") -> None:
    """
    Args:
        matched_df (pd.DataFrame): DataFrame with matched queries
        version1 (str): Name of the first version
        version2 (str): Name of the second version
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

    # Convert version names to column-friendly format (replace dots with underscores)
    version1_col = version1.replace('.', '_')
    version2_col = version2.replace('.', '_')

    # Use the actual column names from the DataFrame
    col1 = f'ms_{version1_col}'
    col2 = f'ms_{version2_col}'

    # Plot 1: Scatter plot comparing versions performance
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.scatter(plot_df[col1], plot_df[col2],
               alpha=0.7, s=10, color='red')

    # Add diagonal line (y=x) for reference
    min_ms = min(plot_df[col1].min(), plot_df[col2].min())
    max_ms = max(plot_df[col1].max(), plot_df[col2].max())
    ax1.plot([min_ms, max_ms], [min_ms, max_ms], 'k--', alpha=0.5, label='Equal Performance')

    ax1.set_xlabel(f'{version1} Query Time (ms)')
    ax1.set_ylabel(f'{version2} Query Time (ms)')
    ax1.set_title(f'Query Performance Comparison: {version1} vs {version2}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/query_performance_comparison-{version1}.png', dpi=300, bbox_inches='tight')
    print(f"Query performance comparison saved as 'plots/query_performance_comparison-{version1}.png'")
    plt.close(fig1)

    # Plot 2: Performance ratio distribution (cap extreme values for better visualization)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ratio_data = plot_df['performance_ratio']
    # Cap ratios at 97th percentile for better visualization
    ratio_cap = ratio_data.quantile(0.97)
    ratio_capped = ratio_data.clip(upper=ratio_cap)

    ax2.hist(ratio_capped, bins=30, alpha=0.7,
             color='purple', edgecolor='black')
    ax2.set_xlabel(f'Performance Ratio ({version2} / {version1}) [capped at {ratio_cap:.1f}x/97th percentile]')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of Performance Ratios: {version1} vs {version2}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/performance_ratio_distribution-{version1}.png', dpi=300, bbox_inches='tight')
    print(f"Performance ratio distribution saved as 'plots/performance_ratio_distribution-{version1}.png'")
    plt.close(fig2)
