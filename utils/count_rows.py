import pandas as pd
import os
from datetime import datetime

# iterate over csv files in "sources" folder and count rows in each file
def count_rows(folder_path):
    """
    Count the number of rows in each CSV file in the specified folder.
    """
    # Initialize dictionary to store row counts
    row_counts = {}
    
    # Iterate over files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            row_counts[file_name] = len(df)
        
    return row_counts

# Example usage
folder_path = "clean"
row_counts = count_rows(folder_path)
for file, count in row_counts.items():
    print(f"File: {file}, Rows: {count}")
