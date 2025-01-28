# Building Energy Analysis Project

This repository contains code and analysis for clustering and analyzing temperature data from building rooms to understand thermal patterns and energy efficiency.

## Project Overview
This project analyzes temperature sensor data from different rooms in a building to identify patterns, group similar rooms together, and provide insights for optimizing energy usage and thermal comfort.

## Repository Structure

### Main Files

- `clustering.py`: Contains functions for loading, preprocessing and clustering room temperature data. Implements K-means clustering to group rooms based on their thermal characteristics and physical properties like volume.

- `data_cleaning.py`: Handles data preprocessing and cleaning of raw sensor data. Removes outliers, handles missing values, and prepares data for analysis.

- `visualization.py`: Creates visualizations of temperature patterns, clustering results, and other relevant insights using matplotlib and seaborn.

- `analysis.py`: Performs statistical analysis on the temperature data and clustering results to extract meaningful insights.

### Key Features

- Data preprocessing and cleaning
- K-means clustering of rooms based on temperature patterns
- Visualization of thermal behavior
- Statistical analysis of temperature distributions
- Integration with room metadata (volume, location, etc.)

### Requirements

The project requires Python 3.7+ and the following main dependencies:
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

See `requirements.txt` for a complete list of dependencies.
