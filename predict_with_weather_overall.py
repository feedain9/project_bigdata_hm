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

# Préparer les données pour les modèles
X = data_merged[["hour", "day_of_week", "temperature_2m"]]  # Variables explicatives
y = data_merged["temperature"]  # Variable cible

# Diviser les données en training et test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Régression linéaire
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

# Évaluer le modèle de régression linéaire
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print(f"Linear Regression - R² Score: {r2_lr:.3f}, MAE: {mae_lr:.3f} °C")

# Visualisation des prédictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.6, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.title("Linear Regression: Actual vs Predicted Temperatures")
plt.xlabel("Actual Temperatures (°C)")
plt.ylabel("Predicted Temperatures (°C)")
plt.grid(True)
plt.show()


# Entraîner un modèle Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Évaluer le modèle Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"Random Forest - R² Score: {r2_rf:.3f}, MAE: {mae_rf:.3f} °C")

# Visualisation des prédictions Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color="green")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.title("Random Forest: Actual vs Predicted Temperatures")
plt.xlabel("Actual Temperatures (°C)")
plt.ylabel("Predicted Temperatures (°C)")
plt.grid(True)
plt.show()
