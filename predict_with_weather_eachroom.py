import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Charger les données des salles et des températures météo
salle_data = pd.read_csv("average_temp_per_room.csv")  # Température des salles
meteo_data = pd.read_csv("open-meteo-48.12N11.40E534m.csv")

# Convertir les colonnes 'time' en datetime pour fusionner les datasets
salle_data["phenomenon_time"] = pd.to_datetime(salle_data["phenomenon_time"])
meteo_data["time"] = pd.to_datetime(meteo_data["time"]).dt.tz_localize('UTC')

# Fusionner les datasets sur la base du temps
data_merged = pd.merge(salle_data, meteo_data, left_on="phenomenon_time", right_on="time")
data_merged.drop(columns=["time"], inplace=True)

# Feature engineering : Extraire les variables temporelles
data_merged["hour"] = data_merged["phenomenon_time"].dt.hour
data_merged["day_of_week"] = data_merged["phenomenon_time"].dt.dayofweek

results = {"Room": [], "R² Score (LR)": [], "MAE (LR)": [], "R² Score (RF)": [], "MAE (RF)": []}

# Liste unique des salles
rooms = data_merged["Room_x"].unique()

for room in rooms:
    print(f"Training models for Room: {room}")
    # Filtrer les données pour la salle spécifique
    room_data = data_merged[data_merged["Room_x"] == room]

    # Variables explicatives et cible
    X = room_data[["hour", "day_of_week", "temperature_2m"]]
    y = room_data["temperature"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Régression linéaire
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)

    # Stocker les résultats
    results["Room"].append(room)
    results["R² Score (LR)"].append(r2_lr)
    results["MAE (LR)"].append(mae_lr)
    results["R² Score (RF)"].append(r2_rf)
    results["MAE (RF)"].append(mae_rf)

# Afficher le résumé des performances
results_df = pd.DataFrame(results)
print(results_df)

# Sauvegarder les résultats dans un fichier CSV
results_df.to_csv("model_performance_per_room_withweather.csv", index=False)
