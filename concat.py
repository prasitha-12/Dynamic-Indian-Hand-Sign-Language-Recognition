import numpy as np
import pandas as pd

# Load the two CSV files excluding headers
data1 = pd.read_csv("features_csv_files/ResNet101_features_LSTM.csv", header=None) # Need to give the path to the file
data2 = pd.read_csv("features_csv_files/Normalized_VLDSPFeatures2.csv", header=None) # Need to give the path to the file

# To ensure both data frames have the same number of rows
min_rows = min(len(data1), len(data2))
data1 = data1[:min_rows]
data2 = data2[:min_rows]

# Convert the data frames to NumPy arrays
array1 = data1.to_numpy()
array2 = data2.to_numpy()

# Concatenate the arrays horizontally using np.hstack
concatenated_array = np.hstack((array1, array2))

# Convert the concatenated array back to a DataFrame
concatenated_df = pd.DataFrame(concatenated_array)

# Save the concatenated DataFrame to a new CSV file
concatenated_df.to_csv("features_csv_files/concatenated_file.csv", index=False, header=False)

print("Concatenated data saved to 'concatenated_file.csv'")