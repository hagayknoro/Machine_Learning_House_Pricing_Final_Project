from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import RobustScaler

def linear_regression_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the data
    scaler = RobustScaler()  # Using RobustScaler to handle outliers
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models with tuned parameters
    models = {
        'Lasso': LassoCV(
            cv=5, 
            random_state=42, 
            max_iter=5000,
            selection='random'
        ),
        'Ridge': RidgeCV(
            alphas=np.logspace(-4, 4, 100)
        ),
        'ElasticNet': ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            cv=5
        )
    }
    
    best_r2 = -float('inf')
    best_model = None
    best_predictions = None
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        current_r2 = r2_score(y_test, predictions)
        
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_model = model
            best_predictions = predictions
    
    # Calculate metrics (return to original scale)
    mse = mean_squared_error(y_test, best_predictions)
    rmse = np.sqrt(mse)
    
    print("\nLinear Regression Details:")
    print(f"Best model type: {type(best_model).__name__}")
    if hasattr(best_model, 'coef_'):
        print(f"Number of features used: {np.sum(best_model.coef_ != 0)}")
    print(f"Cross-validation score: {np.mean(cross_val_score(best_model, X_train_scaled, y_train, cv=5)):.3f}")
    
    return rmse, best_r2

if __name__ == '__main__':
    from data_preprocessing import preprocess_data
    features, target = preprocess_data('Housing.csv')
    print(linear_regression_model(features, target))