from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np

def knn_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2, 3],
        'leaf_size': [10, 30, 50]
    }
    
    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(
        knn, 
        param_grid, 
        cv=5, 
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    predictions = grid_search.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print("\nKNN Details:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    return rmse, r2

if __name__ == '__main__':
    from data_preprocessing import preprocess_data
    features, target = preprocess_data('Housing.csv')
    print(knn_model(features, target))