import pandas as pd
import os
import matplotlib.pyplot as plt

# Define paths and variables
clean_folder = 'clean'  # Folder containing cleaned CSV files
combined_csv_path = 'combined_clean_data.csv'  # File to store combined data
images_folder = 'images'  # Folder to store visualizations

# Combine all cleaned CSV files into one
def combine_cleaned_csv(folder, output_file):
    """
    Combine all cleaned CSV files into a single file.
    """
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    combined_df = pd.concat([pd.read_csv(file) for file in all_files], ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    return combined_df

# Create visualizations for trends and distributions
def create_visualizations(df, output_folder):
    """
    Create visualizations for trends and distributions for each sensor type.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    
    # Loop through each unique sensor type
    for sensor in df['name'].unique():
        sensor_data = df[df['name'] == sensor]
        
        # Convert phenomenonTime to datetime (handle errors)
        sensor_data['phenomenonTime'] = pd.to_datetime(sensor_data['phenomenonTime'], errors='coerce')
        
        # Drop rows with invalid or missing times
        sensor_data = sensor_data.dropna(subset=['phenomenonTime'])

        # Sort data by time
        sensor_data = sensor_data.sort_values(by='phenomenonTime')
        
        # Generate trend visualization
        plt.figure(figsize=(10, 6))
        plt.plot(sensor_data['phenomenonTime'], sensor_data['result'], marker='o', linestyle='-', markersize=2)
        plt.title(f'Trend for {sensor}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.tight_layout()
        trend_path = os.path.join(output_folder, f'trend_{sensor.replace(" ", "_")}.png')
        plt.savefig(trend_path)
        plt.close()
        
        # Generate distribution visualization
        plt.figure(figsize=(10, 6))
        plt.hist(sensor_data['result'], bins=50, alpha=0.7, color='blue')
        plt.title(f'Distribution of {sensor}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        distribution_path = os.path.join(output_folder, f'distribution_{sensor.replace(" ", "_")}.png')
        plt.savefig(distribution_path)
        plt.close()

def create_visualizations_by_sensor(clean_folder, output_folder):
    """
    Create visualizations for trends and distributions for each sensor (per file).
    """
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    
    # Loop through each cleaned CSV file
    for file in os.listdir(clean_folder):
        if file.endswith('.csv'):
            sensor_id = file.split('.')[0]  # Extract sensor ID from file name
            sensor_data = pd.read_csv(os.path.join(clean_folder, file))

            # Filter only Room Temperature data
            sensor_data = sensor_data[sensor_data['name'] == 'Room Temperature']
            
            # Convert phenomenonTime to datetime
            sensor_data['phenomenonTime'] = pd.to_datetime(sensor_data['phenomenonTime'], errors='coerce')
            sensor_data = sensor_data.dropna(subset=['phenomenonTime'])  # Drop rows with invalid times
            sensor_data = sensor_data.sort_values(by='phenomenonTime')  # Sort by time

            # Trend visualization
            plt.figure(figsize=(10, 6))
            plt.plot(sensor_data['phenomenonTime'], sensor_data['result'], marker='o', linestyle='-', markersize=2)
            plt.title(f'Trend for Temperature (Sensor {sensor_id})')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.tight_layout()
            trend_path = os.path.join(output_folder, f'trend_Temperature_{sensor_id}.png')
            plt.savefig(trend_path)
            plt.close()
            
            # Distribution visualization
            plt.figure(figsize=(10, 6))
            plt.hist(sensor_data['result'], bins=50, alpha=0.7, color='blue')
            plt.title(f'Distribution of Temperature (Sensor {sensor_id})')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.tight_layout()
            distribution_path = os.path.join(output_folder, f'distribution_Temperature_{sensor_id}.png')
            plt.savefig(distribution_path)
            plt.close()

# Main pipeline
if __name__ == "__main__":
    # Step 1: Combine cleaned CSV files
    # print(f"Combining cleaned CSV files into '{combined_csv_path}'...")
    # combined_data = combine_cleaned_csv(clean_folder, combined_csv_path)
    
    # # Step 2: Create visualizations
    # print(f"Creating visualizations in '{images_folder}'...")
    # create_visualizations(combined_data, images_folder)

    # print("All tasks completed!")

    create_visualizations_by_sensor(clean_folder='clean', output_folder='images_by_sensor')