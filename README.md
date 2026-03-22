# Boston & California Housing Price Predictor 🏠

A modular Machine Learning project built with Python to predict residential property prices. This project demonstrates a clean, professional approach to data science workflows.

## Project Overview
The goal of this project is to predict the `median_house_value` using historical housing data. By analyzing various features, the model identifies the strongest economic drivers for property costs.

## Modular Architecture
Unlike a simple script, this project is divided into specialized modules for better maintainability and clarity:

* **`data_loader.py`**: Handles data ingestion and cleaning (handling missing values).
* **`explorer.py`**: Performs Exploratory Data Analysis (EDA) and generates correlation heatmaps.
* **`model_trainer.py`**: Manages data splitting (Train/Test) and trains the Linear Regression model.
* **`main.py`**: The central orchestrator that runs the entire pipeline.

## Key Insights
* **Top Predictor**: Through the Correlation Matrix (Heatmap), we identified that **Median Income** has the strongest positive correlation (0.69) with house values.
* **Model Performance**: The Linear Regression model provides a baseline for price estimation based on economic factors.

##Tools & Libraries
* **Language**: Python
* **Data Handling**: Pandas, NumPy
* **Visualization**: Seaborn, Matplotlib
* **Machine Learning**: Scikit-Learn
