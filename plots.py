import matplotlib.pyplot as plt
import polars as pl

def plot_matching_queries(matched_df: pl.DataFrame, version1: str = "Version-1", version2: str = "Version-2") -> None:
    """
    Args:
        matched_df (pl.DataFrame): DataFrame with matched queries
        version1 (str): Name of the first version
        version2 (str): Name of the second version
    """
    if len(matched_df) == 0:
        print("No matching queries found with performance regressions.")
        return

    # Filter out any infinite or NaN values that might cause plotting issues
    plot_df = matched_df.filter(
        pl.col('performance_ratio').is_not_null() &
        pl.col('performance_ratio').is_finite()
    )

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

    # Create single scatter plot with log scales
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert to numpy arrays for matplotlib
    col1_values = plot_df.select(pl.col(col1)).to_numpy().flatten()
    col2_values = plot_df.select(pl.col(col2)).to_numpy().flatten()

    ax.scatter(col1_values, col2_values,
               alpha=0.7, s=10, color='red')

    # Set log base 2 scale on both axes
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    # Add diagonal line (y=x) for reference
    min_ms = min(col1_values.min(), col2_values.min())
    max_ms = max(col1_values.max(), col2_values.max())
    ax.plot([min_ms, max_ms], [min_ms, max_ms], 'k--', alpha=0.5, label='Equal Performance')

    ax.set_xlabel(f'{version1} Query Time (ms) [log₂ scale]')
    ax.set_ylabel(f'{version2} Query Time (ms) [log₂ scale]')
    ax.set_title(f'Query Performance Comparison: {version1} vs {version2}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/query_performance_comparison-{version1}.png', dpi=300, bbox_inches='tight')
    print(f"Query performance comparison saved as 'plots/query_performance_comparison-{version1}.png'")
    plt.close(fig)
