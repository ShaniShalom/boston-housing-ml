import pandas as pd

def load_modern_housing_data():
    """Loads modern housing data from a public CSV file."""
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    try:
        df = pd.read_csv(url)
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
