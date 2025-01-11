import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Charger le dataset
file_path = "average_temp_per_room.csv"  # Remplacer par le chemin du fichier
data = pd.read_csv(file_path)

# Convertir phenomenon_time en datetime et créer des variables temporelles
data["phenomenon_time"] = pd.to_datetime(data["phenomenon_time"])
data["hour"] = data["phenomenon_time"].dt.hour
data["day_of_week"] = data["phenomenon_time"].dt.dayofweek

# Initialiser un dictionnaire pour stocker les performances
results = {"Room": [], "R² Score": [], "MAE": []}

# Liste unique des salles
rooms = data["Room_x"].unique()

# Modélisation pour chaque salle
for room in rooms:
    print(f"Training model for Room: {room}")
    # Filtrer les données par salle
    room_data = data[data["Room_x"] == room]
    
    # Définir les variables explicatives et cible
    X = room_data[["hour", "day_of_week"]]
    y = room_data["temperature"]
    
    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraîner le modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Évaluer le modèle
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Stocker les résultats
    results["Room"].append(room)
    results["R² Score"].append(r2)
    results["MAE"].append(mae)
    
    # Afficher le graphique pour cette salle
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f"Actual vs Predicted Temperatures - Room {room}")
    plt.xlabel("Actual Temperatures (°C)")
    plt.ylabel("Predicted Temperatures (°C)")
    plt.grid()
    plt.show()

# Résumé des performances par salle
results_df = pd.DataFrame(results)
print("\nModel Performance by Room:")
print(results_df)

# Sauvegarder les résultats dans un fichier CSV
results_df.to_csv("model_performance_by_room.csv", index=False)