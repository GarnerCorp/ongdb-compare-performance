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

    # Determine colors based on which version regressed
    # Blue: version1 is slower (col1 > col2, points below diagonal)
    # Red: version2 is slower (col2 > col1, points above diagonal)
    # Green: equal performance (col1 == col2, points on diagonal)
    colors = []
    for i in range(len(col1_values)):
        if col1_values[i] > col2_values[i]:
            colors.append('blue')  # Version 1 regressed
        elif col2_values[i] > col1_values[i]:
            colors.append('red')   # Version 2 regressed
        else:
            colors.append('green') # Equal performance

    # Create scatter plots with different colors
    v1_regression_mask = np.array(colors) == 'blue'
    v2_regression_mask = np.array(colors) == 'red'
    equal_performance_mask = np.array(colors) == 'green'

    if np.any(v1_regression_mask):
        ax.scatter(col1_values[v1_regression_mask], col2_values[v1_regression_mask],
                   alpha=0.7, s=5, color='blue', label=f'{version1} Regressions')

    if np.any(v2_regression_mask):
        ax.scatter(col1_values[v2_regression_mask], col2_values[v2_regression_mask],
                   alpha=0.7, s=5, color='red', label=f'{version2} Regressions')

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
            (pl.col('performance_ratio') <= 16)  # Cap at 16
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
            (pl.col('performance_ratio') <= 16)  # Cap at 16
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
    min_ratio = max(1.0, min(all_ratios))  # Start from 1.0 (no regression)
    max_ratio = min(16.0, max(all_ratios))  # Cap at 16

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

    # Track which labels have been added
    v1_labeled = False
    v2_labeled = False
    equal_labeled = False

    # Determine which bars to plot and in what order
    for i in range(len(bin_centers)):
        v1_count = v1_counts[i]
        v2_count = v2_counts[i]

        if v1_count == 0 and v2_count == 0:
            continue
        elif v1_count == v2_count and v1_count > 0:
            # Same height - plot green bar
            ax.bar(bin_centers[i], v1_count, width=bin_widths[i],
                   color='green', edgecolor='black', linewidth=1,
                   label='Equal Regression Counts' if not equal_labeled else "")
            equal_labeled = True
        else:
            # Different heights - plot taller bar first (behind), then shorter bar
            if v1_count > v2_count:
                # v1 is taller, plot it first (behind)
                if v1_count > 0:
                    ax.bar(bin_centers[i], v1_count, width=bin_widths[i],
                           color='blue', edgecolor='black', linewidth=1,
                           label=f'{version1} Regression Counts (n={len(v1_ratios)})' if not v1_labeled else "")
                    v1_labeled = True
                if v2_count > 0:
                    ax.bar(bin_centers[i], v2_count, width=bin_widths[i],
                           color='red', edgecolor='black', linewidth=1,
                           label=f'{version2} Regression Counts (n={len(v2_ratios)})' if not v2_labeled else "")
                    v2_labeled = True
            else:
                # v2 is taller, plot it first (behind)
                if v2_count > 0:
                    ax.bar(bin_centers[i], v2_count, width=bin_widths[i],
                           color='red', edgecolor='black', linewidth=1,
                           label=f'{version2} Regression Counts (n={len(v2_ratios)})' if not v2_labeled else "")
                    v2_labeled = True
                if v1_count > 0:
                    ax.bar(bin_centers[i], v1_count, width=bin_widths[i],
                           color='blue', edgecolor='black', linewidth=1,
                           label=f'{version1} Regression Counts (n={len(v1_ratios)})' if not v1_labeled else "")
                    v1_labeled = True

    # Set log base 2 scale on both axes
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)


    # Add vertical lines at common regression thresholds
    for threshold, label in [(2.0, '2x slower'), (4.0, '4x slower'), (8.0, '8x slower')]:
        if threshold <= max_ratio:
            ax.axvline(x=threshold, color='gray', linestyle=':', alpha=0.5, label=f'{label}')

    ax.set_xlabel('Performance Ratio (slower_version / faster_version) [log₂ scale, capped at 16x]')
    ax.set_ylabel('Number of Queries [log₂ scale]')
    ax.set_title(f'Distribution of Performance Regression Counts: {version1} vs {version2}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text box with summary statistics
    summary_text = []
    if len(v1_ratios) > 0:
        summary_text.append(f'{version1} regressions: {len(v1_ratios)}')
        summary_text.append(f'  Median ratio: {np.median(v1_ratios):.2f}x')
        summary_text.append(f'  Max ratio: {np.max(v1_ratios):.2f}x')

    if len(v2_ratios) > 0:
        summary_text.append(f'{version2} regressions: {len(v2_ratios)}')
        summary_text.append(f'  Median ratio: {np.median(v2_ratios):.2f}x')
        summary_text.append(f'  Max ratio: {np.max(v2_ratios):.2f}x')

    if summary_text:
        ax.text(0.80, 0.98, '\n'.join(summary_text), transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'plots/regression_histogram-{version1}-vs-{version2}.png', dpi=300, bbox_inches='tight')
    print(f"Regression histogram saved as 'plots/regression_histogram-{version1}-vs-{version2}.png'")
    plt.close(fig)
