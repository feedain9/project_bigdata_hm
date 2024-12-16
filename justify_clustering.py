import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Charger les données depuis le fichier CSV
file_path = "clustering_results.csv"
df = pd.read_csv(file_path)

# Sélectionner les variables pour le clustering
X = df[['R² Score (RF)', 'MAE (RF)', 'Volume']]

# Normaliser les données pour le clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialiser les listes pour stocker les métriques
inertia = []
silhouette_scores = []
k_range = range(2, 6)  # Tester k=2 à k=5

# Calculer l'inertie et le Silhouette Score pour chaque k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Tracer la méthode du coude
plt.figure(figsize=(10, 5))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# Afficher les Silhouette Scores
print("Silhouette Scores for each k:")
for i, k in enumerate(k_range):
    print(f"k = {k}: Silhouette Score = {silhouette_scores[i]:.3f}")

# Optionnel : Tracer les Silhouette Scores
plt.figure(figsize=(10, 5))
plt.plot(k_range, silhouette_scores, marker='o', linestyle='--', color='orange')
plt.title("Silhouette Scores for Different Numbers of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid()
plt.show()