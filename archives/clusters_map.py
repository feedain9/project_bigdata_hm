import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger les données
file_path = "clustering_results.csv"
df = pd.read_csv(file_path)

# Générer des positions X et Y fictives pour chaque salle (si pas de plan disponible)
np.random.seed(42)
df['X'] = np.random.uniform(0, 100, size=len(df))
df['Y'] = np.random.uniform(0, 100, size=len(df))

# Couleurs pour les clusters
cluster_colors = {0: 'purple', 1: 'teal', 2: 'yellow'}

# Tracer la carte des clusters
plt.figure(figsize=(10, 8))
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['X'], cluster_data['Y'], 
                s=cluster_data['Volume'] / 10,  # Taille des points proportionnelle au volume
                color=cluster_colors[cluster], 
                alpha=0.7, label=f"Cluster {cluster}")

# Ajouter des labels pour chaque salle
for i, row in df.iterrows():
    plt.text(row['X'], row['Y'], row['Room'], fontsize=8, ha='center', va='center')

# Ajouter des titres et légendes
plt.title("Abstract Map of Clusters Based on Random Forest Model Performance")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend(title="Cluster")
plt.grid()
plt.show()