import polars as pl
import re
import argparse
from datetime import datetime
from plots import  plot_matching_queries
import os


def extract_timestamp_and_ms(csv_file_path: str) -> pl.DataFrame:
    """
    Extract timestamp and milliseconds from the textPayload column of a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file

    Returns:
        pl.DataFrame: DataFrame with columns 'timestamp', 'milliseconds', and 'original_text'
    """
    df = pl.read_csv(csv_file_path)

    if 'textPayload' not in df.columns:
        raise ValueError("textPayload column not found in CSV file")

    timestamps = []
    milliseconds = []
    original_texts = []

    # Regular expression pattern to match timestamp and milliseconds
    # Pattern: '2025-08-13 11:59:58.946+0000 INFO  0 ms:
    pattern = r"'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\+\d{4}) INFO\s+(\d+) ms:"

    for idx, text_payload in enumerate(df['textPayload']):
        if text_payload is None:
            timestamps.append(None)
            milliseconds.append(None)
            original_texts.append(None)
            continue

        match = re.search(pattern, str(text_payload))

        if match:
            timestamp_str = match.group(1)
            ms_value = int(match.group(2))

            try:
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
            timestamps.append(None)
            milliseconds.append(None)
            original_texts.append(text_payload)

    result_df = pl.DataFrame({
        'timestamp': timestamps,
        'milliseconds': milliseconds,
        'original_text': original_texts
    })

    return result_df

def extract_query_text(text_payload: str) -> str:
    """
    Extract the query part from textPayload (everything after "> neo4j - ").

    Args:
        text_payload (str): The full textPayload string

    Returns:
        str: The extracted query text, or empty string if not found
    """
    if text_payload is None:
        return ""

    # Find the pattern "> neo4j - " and extract everything after it
    pattern = r'>\s*neo4j\s+- (.+)'
    match = re.search(pattern, str(text_payload))
    # print(f"Extracting query from textPayload: {text_payload[100:300]}...")  # Debug output

    if match:
        return match.group(1).strip()
    return ""

def find_all_matching_queries(df1: pl.DataFrame, df2: pl.DataFrame,
                         time_threshold_ms: int = 10000) -> pl.DataFrame:
    """
    Find all queries that appear in both datasets within a time window (not just regressions).

    Args:
        df1 (pl.DataFrame): DataFrame for version 1 logs
        df2 (pl.DataFrame): DataFrame for version 2 logs
        time_threshold_ms (int): Time window in milliseconds for matching queries

    Returns:
        pl.DataFrame: DataFrame with all matched queries and their performance comparison
    """
    # Filter valid entries
    df1_valid = df1.filter(
        pl.col('timestamp').is_not_null() &
        pl.col('milliseconds').is_not_null() &
        pl.col('original_text').is_not_null()
    ).clone()
    df2_valid = df2.filter(
        pl.col('timestamp').is_not_null() &
        pl.col('milliseconds').is_not_null() &
        pl.col('original_text').is_not_null()
    ).clone()

    # Extract query text for both datasets
    df1_valid = df1_valid.with_columns(
        pl.col('original_text').map_elements(extract_query_text, return_dtype=pl.String).alias('query_text')
    )
    df2_valid = df2_valid.with_columns(
        pl.col('original_text').map_elements(extract_query_text, return_dtype=pl.String).alias('query_text')
    )

    # Remove entries with empty query text
    df1_valid = df1_valid.filter(pl.col('query_text') != "")
    df2_valid = df2_valid.filter(pl.col('query_text') != "")

    matched_queries = []
    used_df2_indices = set()  # Track which df2 entries we've already matched

    print(f"Searching for matches in {len(df1_valid)} df1 entries and {len(df2_valid)} df2 entries...")

    # Convert to row dictionaries for iteration
    df1_rows = df1_valid.to_dicts()
    df2_rows = df2_valid.to_dicts()

    for row1 in df1_rows:
        query1 = row1['query_text']
        timestamp1 = row1['timestamp']
        ms1 = row1['milliseconds']

        best_match = None
        best_time_diff = float('inf')

        # Find the best matching query in df2 (closest in time)
        for idx2, row2 in enumerate(df2_rows):
            if idx2 in used_df2_indices:
                continue  # Skip already matched entries

            query2 = row2['query_text']
            timestamp2 = row2['timestamp']
            ms2 = row2['milliseconds']

            # Check if queries match exactly
            if query1 == query2:
                time_diff = abs((timestamp2 - timestamp1).total_seconds() * 1000)

                # Consider all matches within time threshold (not just regressions)
                if time_diff <= time_threshold_ms:
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
                            'performance_ratio': ms2 / ms1 if ms1 > 0 else float('inf')
                        }
                        best_time_diff = time_diff

        # Add the best match if found
        if best_match:
            used_df2_indices.add(best_match['idx2'])
            del best_match['idx2']  # Remove internal tracking field
            matched_queries.append(best_match)

    print(f"Found {len(matched_queries)} unique matching queries")
    return pl.DataFrame(matched_queries)

def get_matching_query_stats(matched_df: pl.DataFrame, version_col: str) -> dict:
    """
    Get performance statistics for matching queries from a specific version column.

    Args:
        matched_df (pl.DataFrame): DataFrame with matched queries
        version_col (str): Column name for the version to analyze ('ms_1_0_6' or 'ms_1_1_0')

    Returns:
        dict: Dictionary with performance statistics
    """
    if len(matched_df) == 0:
        return {"error": "No matched queries found"}

    values = matched_df.select(pl.col(version_col))

    stats = {
        "total_matching_queries": len(matched_df),
        "avg_ms": values.select(pl.col(version_col).mean()).item(row=0, column=0),
        "median_ms": values.select(pl.col(version_col).median()).item(row=0, column=0),
        "min_ms": values.select(pl.col(version_col).min()).item(row=0, column=0),
        "max_ms": values.select(pl.col(version_col).max()).item(row=0, column=0),
        "std_ms": values.select(pl.col(version_col).std()).item(row=0, column=0)
    }

    return stats

def analyze_query_regressions(matched_df: pl.DataFrame, version1: str = "Version 1", version2: str = "Version 2") -> dict:
    """
    Analyze the query performance regressions and return summary statistics.

    Args:
        matched_df (pl.DataFrame): DataFrame with matched queries
        version1 (str): Name of the first version
        version2 (str): Name of the second version

    Returns:
        dict: Dictionary with regression analysis statistics
    """
    if len(matched_df) == 0:
        return {"error": "No matched queries found"}

    # Filter out infinite values for meaningful statistics
    valid_ratios_df = matched_df.filter(
        pl.col('performance_ratio').is_not_null() &
        pl.col('performance_ratio').is_finite()
    )

    analysis = {
        "total_matched_queries": len(matched_df),
        "avg_regression_ms": matched_df.select(pl.col('performance_regression').mean()).item(row=0, column=0),
        "median_regression_ms": matched_df.select(pl.col('performance_regression').median()).item(row=0, column=0),
        "max_regression_ms": matched_df.select(pl.col('performance_regression').max()).item(row=0, column=0),
        "min_regression_ms": matched_df.select(pl.col('performance_regression').min()).item(row=0, column=0),
        "avg_performance_ratio": valid_ratios_df.select(pl.col('performance_ratio').mean()).item(row=0, column=0) if len(valid_ratios_df) > 0 else 0,
        "median_performance_ratio": valid_ratios_df.select(pl.col('performance_ratio').median()).item(row=0, column=0) if len(valid_ratios_df) > 0 else 0,
        "max_performance_ratio": valid_ratios_df.select(pl.col('performance_ratio').max()).item(row=0, column=0) if len(valid_ratios_df) > 0 else 0,
        "worst_regression_query": matched_df.sort('performance_regression', descending=True).select(pl.col('query_text')).item(row=0, column=0)[:100] + "..." if len(matched_df) > 0 else "",
        "queries_with_2x_slowdown": len(matched_df.filter(pl.col('performance_ratio') >= 2.0)),
        "queries_with_5x_slowdown": len(matched_df.filter(pl.col('performance_ratio') >= 5.0)),
        "queries_with_10x_slowdown": len(matched_df.filter(pl.col('performance_ratio') >= 10.0))
    }

    return analysis

def printItems(items: dict):
    for key, value in items.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare database log performance between two versions')
    parser.add_argument('version1', help='Database version 1 (e.g., "1.0.6")')
    parser.add_argument('logs1', help='Path to log file 1')
    parser.add_argument('version2', help='Database version 2 (e.g., "1.1.0")')
    parser.add_argument('logs2', help='Path to log file 2')

    args = parser.parse_args()
    os.makedirs('plots', exist_ok=True)


    try:
        df1 = extract_timestamp_and_ms(args.logs1)
        df2 = extract_timestamp_and_ms(args.logs2)

        print(f"Extracted {len(df1)} entries from {args.logs1}")
        print(f"Valid entries with timestamp/ms: {len(df1.filter(pl.col('milliseconds').is_not_null()))}")

        print(f"Extracted {len(df2)} entries from {args.logs2}")
        print(f"Valid entries with timestamp/ms: {len(df2.filter(pl.col('milliseconds').is_not_null()))}")

        # Filter out 0ms entries and report counts
        df1_valid = df1.filter(pl.col('milliseconds').is_not_null())
        df2_valid = df2.filter(pl.col('milliseconds').is_not_null())

        df1_zero_count = len(df1_valid.filter(pl.col('milliseconds') == 0))
        df2_zero_count = len(df2_valid.filter(pl.col('milliseconds') == 0))

        df1_filtered = df1_valid.filter(pl.col('milliseconds') > 0)
        df2_filtered = df2_valid.filter(pl.col('milliseconds') > 0)

        print(f"\nFiltered out {df1_zero_count} entries with 0ms from {args.version1}")
        print(f"Remaining entries for {args.version1}: {len(df1_filtered)}")
        print(f"\nFiltered out {df2_zero_count} entries with 0ms from {args.version2}")
        print(f"Remaining entries for {args.version2}: {len(df2_filtered)}")

        # Find all matching queries (not just regressions)
        print("\nFinding all matching queries...")
        all_matched_queries = find_all_matching_queries(df1_filtered, df2_filtered, time_threshold_ms=10000)

        if len(all_matched_queries) > 0:
            same_time_queries = all_matched_queries.filter(pl.col('ms_1_0_6') == pl.col('ms_1_1_0'))
            print(f"Queries with exact same timing: {len(same_time_queries)}")

            print(f"Found {len(all_matched_queries)} total matching queries between {args.version1} and {args.version2}")

            # Get performance stats for just the matching queries
            matching_stats1 = get_matching_query_stats(all_matched_queries, version_col='ms_1_0_6')
            matching_stats2 = get_matching_query_stats(all_matched_queries, version_col='ms_1_1_0')

            print(f"\n{args.version1} Performance Stats (Matching Queries Only):")
            printItems(matching_stats1)

            print(f"\n{args.version2} Performance Stats (Matching Queries Only):")
            printItems(matching_stats2)

            # Analyze performance regressions in both directions
            print("\nAnalyzing performance changes in both directions...")
            regressions_v2_slower = all_matched_queries.filter(pl.col('ms_1_1_0') > pl.col('ms_1_0_6'))
            regressions_v1_slower = all_matched_queries.filter(pl.col('ms_1_0_6') > pl.col('ms_1_1_0'))

            print(f"\nQueries where {args.version2} is slower than {args.version1}: {len(regressions_v2_slower)}")
            if len(regressions_v2_slower) > 0:
                regression_analysis_v2 = analyze_query_regressions(regressions_v2_slower, args.version1, args.version2)
                print(f"\n{args.version2} Regression Analysis:")
                printItems(regression_analysis_v2)

            print(f"\nQueries where {args.version1} is slower than {args.version2}: {len(regressions_v1_slower)}")
            if len(regressions_v1_slower) > 0:
                regressions_v1_slower_swapped = regressions_v1_slower.with_columns([
                    (pl.col('ms_1_0_6') - pl.col('ms_1_1_0')).alias('performance_regression'),
                    (pl.col('ms_1_0_6') / pl.col('ms_1_1_0')).alias('performance_ratio')
                ])
                regression_analysis_v1 = analyze_query_regressions(regressions_v1_slower_swapped, args.version2, args.version1)
                print(f"\n{args.version1} Regression Analysis:")
                printItems(regression_analysis_v1)

            print(f"\nGenerating regression analysis graphs for {args.version2} slower cases...")
            plot_matching_queries(all_matched_queries, args.version1, args.version2)

        else:
            print("No matching queries found between the two datasets.")

        print("\nAnalysis complete! Check the generated PNG files for visual comparisons (if plotting functions are available).")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()