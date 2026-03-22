from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_housing_model(df):
    """
    Trains a simple Linear Regression model.
    Predicts 'median_house_value' based on 'median_income'.
    """
    
    # X = the influencing information (income), y = the target (price)
    X = df[['median_income']] 
    y = df['median_house_value']
    
    # Divided into 80% learning and 20% testingן
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model creation and training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Performance testing on test data
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    return model, rmse, r2
