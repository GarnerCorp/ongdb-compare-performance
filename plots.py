import matplotlib.pyplot as plt
import polars as pl
import numpy as np

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

    # Use the actual column names from the DataFrame
    col1 = 'ms_1'
    col2 = 'ms_2'

    # Create single scatter plot with log scales
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert to numpy arrays for matplotlib
    col1_values = plot_df.select(pl.col(col1)).to_numpy().flatten()
    col2_values = plot_df.select(pl.col(col2)).to_numpy().flatten()

    # Determine colors based on which version regressed
    # Blue: version1 is slower (col1 > col2, points below diagonal)
    # Red: version2 is slower (col2 > col1, points above diagonal)
    # Green: equal performance (col1 == col2, points on diagonal)
    colors = []
    for i, col1_value in enumerate(col1_values):
        col2_value = col2_values[i]
        if col1_value > col2_value:
            colors.append('blue')  # Version 1 regressed
        elif col2_value > col1_value:
            colors.append('red')   # Version 2 regressed
        else:
            colors.append('green') # Equal performance

    # Create scatter plots with different colors
    v1_regression_mask = np.array(colors) == 'blue'
    v2_regression_mask = np.array(colors) == 'red'
    equal_performance_mask = np.array(colors) == 'green'

    if np.any(v1_regression_mask):
        ax.scatter(col1_values[v1_regression_mask], col2_values[v1_regression_mask],
                   alpha=0.7, s=5, color='blue', label=f'{version1} Slower')

    if np.any(v2_regression_mask):
        ax.scatter(col1_values[v2_regression_mask], col2_values[v2_regression_mask],
                   alpha=0.7, s=5, color='red', label=f'{version2} Slower')

    if np.any(equal_performance_mask):
        ax.scatter(col1_values[equal_performance_mask], col2_values[equal_performance_mask],
                   alpha=0.7, s=5, color='green', label='Equal Performance')

    # Set log base 2 scale on both axes
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    # Add diagonal line (y=x) for reference
    min_ms = min(col1_values.min(), col2_values.min())
    max_ms = max(col1_values.max(), col2_values.max())
    ax.plot([min_ms, max_ms], [min_ms, max_ms], 'k--', alpha=0.5, linewidth=0.8, label='Equivalent Performance')

    ax.set_xlabel(f'{version1} Query Time (ms) [log₂ scale]')
    ax.set_ylabel(f'{version2} Query Time (ms) [log₂ scale]')
    ax.set_title(f'Query Performance Comparison: {version1} vs {version2}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/query_performance_comparison-{version1}.png', dpi=300, bbox_inches='tight')
    print(f"Query performance comparison saved as 'plots/query_performance_comparison-{version1}.png'")
    plt.close(fig)

def plot_regression_histogram(regressions_v1: pl.DataFrame, regressions_v2: pl.DataFrame,
                            version1: str = "Version-1", version2: str = "Version-2") -> None:
    """
    Create a stacked histogram of performance ratios for queries that regressed in each version.

    Args:
        regressions_v1 (pl.DataFrame): DataFrame with queries where version1 is slower
        regressions_v2 (pl.DataFrame): DataFrame with queries where version2 is slower
        version1 (str): Name of the first version
        version2 (str): Name of the second version
    """
    if len(regressions_v1) == 0 and len(regressions_v2) == 0:
        print("No regressions found in either version for histogram plotting.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract performance ratios and filter out infinite values
    v1_ratios = []
    v2_ratios = []

    if len(regressions_v1) > 0:
        v1_ratios_df = regressions_v1.filter(
            pl.col('performance_ratio').is_not_null() &
            pl.col('performance_ratio').is_finite() &
            (pl.col('performance_ratio') > 0) &
            (pl.col('performance_ratio') <= 16)
        )
        if len(v1_ratios_df) > 0:
            v1_ratios = v1_ratios_df.select(pl.col('performance_ratio')).to_numpy().flatten()
        else:
            print("Debug: v1_ratios_df is empty after filtering")

    if len(regressions_v2) > 0:
        v2_ratios_df = regressions_v2.filter(
            pl.col('performance_ratio').is_not_null() &
            pl.col('performance_ratio').is_finite() &
            (pl.col('performance_ratio') > 0) &
            (pl.col('performance_ratio') <= 16)
        )
        if len(v2_ratios_df) > 0:
            v2_ratios = v2_ratios_df.select(pl.col('performance_ratio')).to_numpy().flatten()
        else:
            print("Debug: v2_ratios_df is empty after filtering")

    if len(v1_ratios) == 0 and len(v2_ratios) == 0:
        print("No valid performance ratios for histogram plotting.")
        return

    # Determine histogram bins based on the data range
    all_ratios = []
    if len(v1_ratios) > 0:
        all_ratios.extend(v1_ratios)
    if len(v2_ratios) > 0:
        all_ratios.extend(v2_ratios)

    # Use log base 2 spaced bins for better visualization of performance ratios
    min_ratio = max(1.0, min(all_ratios))
    max_ratio = min(16.0, max(all_ratios))

    # Create log base 2 spaced bins
    bins = np.logspace(np.log2(min_ratio), np.log2(max_ratio), 30, base=2)

    # Calculate histogram values for both datasets
    v1_counts = np.zeros(len(bins)-1)
    v2_counts = np.zeros(len(bins)-1)

    if len(v1_ratios) > 0:
        v1_counts, _ = np.histogram(v1_ratios, bins=bins)

    if len(v2_ratios) > 0:
        v2_counts, _ = np.histogram(v2_ratios, bins=bins)

    # Create the layered histogram with opaque bars
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = bins[1:] - bins[:-1]

    labeled_flags = {'v1': False, 'v2': False, 'equal': False}
    def plot_bar(count, bar_type, version_name):
        if count <= 0:
            return

        colors = {'v1': 'blue', 'v2': 'red', 'equal': 'green'}

        label = ""
        if bar_type == 'equal' and not labeled_flags['equal']:
            label = 'Equal Slower'
            labeled_flags['equal'] = True
        elif bar_type in ['v1', 'v2'] and not labeled_flags[bar_type]:
            label = f'{version_name} Slower'
            labeled_flags[bar_type] = True

        ax.bar(bin_center, count, width=bin_width, color=colors[bar_type],
               edgecolor='black', linewidth=1, label=label)


    for i, bin_center in enumerate(bin_centers):
        v1_count = v1_counts[i]
        v2_count = v2_counts[i]
        bin_width = bin_widths[i]
        if v1_count == 0 and v2_count == 0:
            continue
        elif v1_count == v2_count and v1_count > 0:
            plot_bar(v1_count, 'equal', version1)
        elif v1_count > v2_count:
            plot_bar(v1_count, 'v1', version1)
            plot_bar(v2_count, 'v2', version2)
        else:
            plot_bar(v2_count, 'v2', version2)
            plot_bar(v1_count, 'v1', version1)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    ax.set_xlabel('Performance Ratio (slower version / faster version) [log₂ scale, capped at 16x]')
    ax.set_ylabel('Number of Queries [log₂ scale]')
    ax.set_title(f'Distribution of Slower Query Counts: {version1} (n={len(v1_ratios)}) vs {version2} (n={len(v2_ratios)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/slower_query_histogram-{version1}-vs-{version2}.png', dpi=300, bbox_inches='tight')
    print(f"Slower query histogram saved as 'plots/slower_query_histogram-{version1}-vs-{version2}.png'")
    plt.close(fig)
