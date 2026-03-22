import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlation_matrix(df):
    """Shows how variables relate to each other."""
    plt.figure(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.show()
