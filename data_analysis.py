import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths to files
clean_combined_csv = 'combined_clean_data.csv'  # Combined cleaned data from the previous steps
classroom_file = 'sources/Classroom_list_FK10.xlsx'  # Excel file with room and sensor mapping

# Load datasets
def load_data(clean_file, classroom_file):
    """
    Load the cleaned temperature data and the classroom mapping file.
    """
    # Load combined cleaned temperature data
    clean_data = pd.read_csv(clean_file)
    
    # Filter only Room Temperature data
    clean_data = clean_data[clean_data['name'] == 'Room Temperature']
    
    # Load classroom mapping
    classroom_data = pd.read_excel(classroom_file)
    
    return clean_data, classroom_data

# Calculate average temperature by room
def calculate_average_temperature(clean_data, classroom_data):
    """
    Merge sensor data with classroom data and calculate average temperature by room.
    """
    # Merge data based on Sensor_ID and thing_id
    merged_data = pd.merge(
        clean_data,
        classroom_data,
        left_on='thing_id',
        right_on='Thing ID',
        how='inner'
    )
    
    # Calculate average temperature per room
    room_avg_temp = merged_data.groupby('Room').agg(
        average_temperature=('result', 'mean'),
        room_volume=('Room Volume', 'first')  # Assuming room volume is consistent per room
    ).reset_index()
    
    return room_avg_temp

# Visualize relationship between room volume and average temperature
def visualize_volume_temperature_relationship(room_avg_temp):
    """
    Create a scatter plot to visualize the relationship between room volume and average temperature.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(room_avg_temp['room_volume'], room_avg_temp['average_temperature'], alpha=0.7)
    plt.title('Room Volume vs Average Temperature')
    plt.xlabel('Room Volume (m³)')
    plt.ylabel('Average Temperature (°C)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('volume_vs_temperature.png')
    plt.show()


# Calculate monthly average temperature by room
def calculate_monthly_average_temperature(clean_data, classroom_data):
    """
    Merge sensor data with classroom data and calculate monthly average temperature by room.
    """
    # Convert phenomenonTime to datetime
    clean_data['phenomenonTime'] = pd.to_datetime(clean_data['phenomenonTime'], errors='coerce')
    
    # Extract month and year for grouping
    clean_data['month'] = clean_data['phenomenonTime'].dt.to_period('M')  # Period for month
    
    # Merge data based on Sensor_ID and Thing ID
    merged_data = pd.merge(
        clean_data,
        classroom_data,
        left_on='thing_id',
        right_on='Thing ID',
        how='inner'
    )
    
    # Calculate monthly average temperature per room
    monthly_avg_temp = merged_data.groupby(['Room', 'month']).agg(
        average_temperature=('result', 'mean')
    ).reset_index()
    
    return monthly_avg_temp

# Create a heatmap
def create_heatmap(monthly_avg_temp):
    """
    Create a heatmap for the monthly average temperature by room.
    """
    # Pivot data for heatmap
    heatmap_data = monthly_avg_temp.pivot(index='Room', columns='month', values='average_temperature')
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        cbar_kws={'label': 'Average Temperature (°C)'},
        linewidths=0.5
    )
    plt.title('Monthly Average Temperature by Room')
    plt.xlabel('Month')
    plt.ylabel('Room')
    plt.tight_layout()
    plt.savefig('monthly_avg_temperature_heatmap.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    clean_data, classroom_data = load_data(clean_combined_csv, classroom_file)
    
    # Calculate average temperature
    room_avg_temp = calculate_average_temperature(clean_data, classroom_data)
    print("Average temperature per room:")
    print(room_avg_temp)
    
    # Visualize the relationship
    visualize_volume_temperature_relationship(room_avg_temp)

    # Calculate monthly average temperature
    monthly_avg_temp = calculate_monthly_average_temperature(clean_data, classroom_data)
    print("Monthly average temperature per room:")
    print(monthly_avg_temp)
    
    # Create heatmap
    create_heatmap(monthly_avg_temp)