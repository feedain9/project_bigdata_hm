# Building Temperature Prediction Project

This repository contains code and analysis for clustering and analyzing temperature data from building rooms to understand thermal patterns and energy efficiency.

## Project Overview
This project analyzes temperature sensor data from different rooms in a building to identify patterns, group similar rooms together, and provide insights for optimizing energy usage and thermal comfort.

## Repository Structure

### Main Files

- `clustering.py`: Contains functions for loading, preprocessing and clustering room temperature data. Implements K-means clustering to group rooms based on their thermal characteristics and physical properties like volume.

- `clean_data.py`: Handles data preprocessing and cleaning of raw sensor data. Removes outliers, handles missing values, and prepares data for analysis.

- `visualisation.py`: Creates visualizations of temperature patterns, clustering results, and other relevant insights using matplotlib and seaborn.

- `data_analysis.py`: Performs statistical analysis on the temperature data and clustering results to extract meaningful insights.

- `predicting.py`: Implements machine learning models to predict room temperatures

### Key Features & Code Files

- `clustering.py`: Performs K-means clustering analysis on room temperature data
  - Loads and preprocesses temperature and room metadata
  - Implements K-means clustering to group similar rooms
  - Finds optimal number of clusters using elbow method and silhouette scores
  - Visualizes clustering results with PCA dimensionality reduction
  - Generates cluster plots and evaluation metrics

- `data_analysis.py`: Statistical analysis of temperature patterns
  - Calculates average temperatures by room and time period
  - Analyzes relationship between room volume and temperature
  - Creates heatmaps of monthly temperature variations
  - Generates statistical insights and visualizations

- `predicting.py` & `predicting_bis.py`: Temperature prediction models
  - Implements Random Forest regression for temperature prediction
  - Handles feature engineering and data preprocessing
  - Evaluates model performance with RMSE and R² metrics
  - Visualizes predictions vs actual values
  - Analyzes feature importance for each room
  - `predicting_bis.py` experiments with different feature combinations (Room Volume alone, and all features + sensor name) compared to `predicting.py`.

- `visualisation.py`: Data visualization and reporting
  - Combines and processes cleaned sensor data
  - Creates trend visualizations for each sensor
  - Generates distribution plots of temperature values
  - Produces time series plots of temperature patterns
  - Saves visualizations to organized folders

The project requires Python 3.7+ and the following dependencies:

### Requirements

The project requires Python 3.7+ and the following main dependencies:
- pandas
- matplotlib
- seaborn

See `requirements.txt` for a complete list of dependencies.