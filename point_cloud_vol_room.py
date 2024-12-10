import matplotlib.pyplot as plt

# Data from the table provided
room_data = {
    "Room": ["LE-001", "LE-002", "LE-003", "LE-006", "LE-007", "LE-008", "LE-015", "LO-101",
             "LO-102", "LO-103", "LO-106", "LO-107", "LO-108", "LO-119", "LO-201", "LO-202",
             "LO-203", "LO-301", "LO-302", "LO-304", "LO-305", "LO-308", "LO-309"],
    "Volume": [311, 314, 303, 232, 287, 232, 3800, 303, 309, 308, 228, 276, 232, 200, 231, 231, 
               224, 150, 153, 151, 301, 229, 274],
    "Temperature": [15.0, 14.3, 14.8, 14.3, 15.1, 15.0, 12.1, 15.7, 16.2, 15.9, 16.1, 16.7, 15.3,
                    16.4, 16.4, 16.9, 17.0, 17.2, 17.4, 17.8, 17.1, 16.7, 17.2]
}

# Scatter plot of Volume vs Temperature
plt.figure(figsize=(10, 6))
plt.scatter(room_data["Volume"], room_data["Temperature"], color='blue', alpha=0.7)

# Add labels and title
plt.title("Relationship Between Room Volume and Average Temperature", fontsize=14)
plt.xlabel("Room Volume (m³)", fontsize=12)
plt.ylabel("Average Temperature (°C)", fontsize=12)
plt.grid(True)
plt.show()
