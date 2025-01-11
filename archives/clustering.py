import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données de performance et ajouter les volumes des salles
data = pd.read_csv("model_performance_per_room_withweather.csv")  # Contient Room, R² (RF), MAE (RF)
volumes = pd.read_csv("Cleaning Project Big Data - Rooms.csv")  # Contient Room, Volume
data = pd.merge(data, volumes, on="Room")

# Sélection des variables pour le clustering
X = data[["R² Score (RF)", "MAE (RF)", "Volume"]]

# Normalisation des données pour le clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer K-Means avec un nombre de clusters optimal (k=3 après tests)
kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(X_scaled)

# Visualiser les clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x="R² Score (RF)", y="MAE (RF)", hue="Cluster", size="Volume", data=data, palette="viridis", sizes=(20, 200))
plt.title("Room clustering based on Random Forest model performance")
plt.xlabel("R² Score (RF)")
plt.ylabel("MAE (RF)")
plt.legend(title="Cluster")
plt.show()

# Afficher les résultats finaux
print(data[["Room", "R² Score (RF)", "MAE (RF)", "Volume", "Cluster"]])

# Sauvegarder le résultat
data.to_csv("clustering_results.csv", index=False)
