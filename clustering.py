import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

# Paths to data
clean_combined_csv = 'clean/combined_clean_data.csv'
classroom_file = 'sources/Classroom_list_FK10.xlsx'

# Load data
def load_data(clean_file, classroom_file):
    """
    Load the cleaned temperature data and the classroom mapping file.
    """
    clean_data = pd.read_csv(clean_file)
    clean_data = clean_data[clean_data['name'] == 'Room Temperature']
    classroom_data = pd.read_excel(classroom_file)
    return clean_data, classroom_data

# Prepare data for clustering
def prepare_clustering_data(clean_data, classroom_data):
    """
    Prepare the data for clustering by calculating room-specific metrics.
    """
    # Merge temperature data with classroom data
    clean_data['phenomenonTime'] = pd.to_datetime(clean_data['phenomenonTime'], errors='coerce')
    merged_data = pd.merge(
        clean_data,
        classroom_data,
        left_on='thing_id',
        right_on='Thing ID',
        how='inner'
    )
    
    # Calculate room-level metrics
    room_metrics = merged_data.groupby('Room').agg(
        average_temperature=('result', 'mean'),
        room_volume=('Room Volume', 'first')  # Assuming room volume is consistent per room
    ).reset_index()
    
    return room_metrics

# Perform clustering
def perform_clustering(room_metrics, n_clusters=3):
    """
    Perform clustering on the room metrics.
    """
    # Standardize the data
    features = ['average_temperature', 'room_volume']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(room_metrics[features])
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    room_metrics['cluster'] = kmeans.fit_predict(scaled_data)
    
    return room_metrics, kmeans, scaled_data

# Visualize clustering results
def visualize_clusters(room_metrics, scaled_data):
    """
    Visualize clustering results using PCA for dimensionality reduction.
    """
    # Reduce dimensions to 2D with PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    
    # Add PCA results to room metrics
    room_metrics['PCA1'] = reduced_data[:, 0]
    room_metrics['PCA2'] = reduced_data[:, 1]
    
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='PCA1',
        y='PCA2',
        hue='cluster',
        palette='Set2',
        data=room_metrics,
        s=100
    )
    plt.title('Clustering of Rooms Based on Temperature and Volume')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig('room_clusters.png')
    plt.show()


# Perform clustering with justification
def find_optimal_clusters(room_metrics, max_clusters=10):
    """
    Find the optimal number of clusters using the Elbow Method and Silhouette Score.
    """
    features = ['average_temperature', 'room_volume']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(room_metrics[features])
    
    inertia = []
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Calculate inertia
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score
        score = silhouette_score(scaled_data, cluster_labels)
        silhouette_scores.append(score)
        print(f"Clusters: {n_clusters} | Silhouette Score: {score:.3f}")
    
    # Plot Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), inertia, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('elbow_method.png')
    plt.show()
    
    # Plot Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='--', color='green')
    plt.title('Silhouette Score for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('silhouette_score.png')
    plt.show()

    return inertia, silhouette_scores

# Main execution
if __name__ == "__main__":
    # Load data
    clean_data, classroom_data = load_data(clean_combined_csv, classroom_file)
    
    # Prepare clustering data
    room_metrics = prepare_clustering_data(clean_data, classroom_data)

    # Justify the number of clusters
    inertia, silhouette_scores = find_optimal_clusters(room_metrics, max_clusters=10)
    
    # Perform clustering
    room_metrics, kmeans, scaled_data = perform_clustering(room_metrics, n_clusters=3)
    
    # Display clustered data
    print("Clustering Results:")
    print(room_metrics)
    
    # Visualize clusters
    visualize_clusters(room_metrics, scaled_data)
