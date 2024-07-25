import pandas as pd

# Load the flight log data
flight_log_path = 'data/Aug-30th-2022-12-59PM-Flight-Airdata.csv'
flight_log_data = pd.read_csv(flight_log_path)

# Print the columns to inspect the column names
print(flight_log_data.columns)

# Filter the data to only include rows where 'isVideo' is 1
filtered_log_data = flight_log_data[flight_log_data['isVideo'] == 1].reset_index(drop=True)

# Print a sample of the filtered data
print(filtered_log_data.head())
