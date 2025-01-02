import pandas as pd

# Function to process and aggregate the dataset
def process_and_aggregate_data(input_file, output_file):
    # Load the dataset
    data = pd.read_csv(input_file, sep=',', quotechar='"')

    # Function to classify render stage (only for CLS, LCP, FCP, FID, INP)
    def classify_render_stage(row):
        if row['metricName'] in ['CLS', 'LCP', 'FCP']:
            if "Initial" in row['scenario']:
                return "primary"
            elif "Re-rendering" in row['scenario']:
                return "secondary"
        return "all"  # For metrics not in these categories

    # Add a new column for render stage
    data['render_stage'] = data.apply(classify_render_stage, axis=1)

    # Add a binary column for level (1 for 'info', 0 for others)
    data['level_info'] = data['level'].apply(lambda x: 1 if x == 'info' else 0)

    # Determine the aggregated value for level_info (1 if all are 1, else 0)
    def aggregate_level_info(group):
        return 1 if all(group == 1) else 0

    # Apply the aggregation function to group by eventId
    level_info_aggregated = data.groupby('eventId')['level_info'].apply(aggregate_level_info)
    
    # Pivot the table to create wide-format data
    aggregated_data = data.pivot_table(
        index='eventId',
        columns=['render_stage', 'metricName'],
        values='value',
        aggfunc='mean'  # Aggregation method (mean)
    )

    # Add the aggregated 'level_info' column to the output
    aggregated_data = aggregated_data.reset_index()
    aggregated_data['level_info'] = aggregated_data['eventId'].map(level_info_aggregated)

    # Flatten MultiIndex columns
    aggregated_data.columns = [
        f"{stage}_{metric}" if stage != "all" else metric
        for stage, metric in aggregated_data.columns
    ]

    # Fill missing values with 0
    aggregated_data = aggregated_data.fillna(0)

    # Save the processed data to a new CSV file
    aggregated_data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

# Paths for the input and output files
input_file_path = './data/log_data_v2.csv'
output_file_path = './processed_data.csv'

# Process the data and save it
process_and_aggregate_data(input_file_path, output_file_path)
