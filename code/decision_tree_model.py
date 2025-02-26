from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import numpy as np

def decision_tree_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'max_depth': [3, 4, 5, 6],  # Smaller depth
        'min_samples_split': [5, 10, 15],  # Larger values
        'min_samples_leaf': [3, 5, 7],  # Larger values
        'max_features': ['sqrt'],  # Fewer options
        'criterion': ['squared_error', 'friedman_mse']
    }
    
    # Create model with GridSearch
    dt = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(
        dt,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    # Train the model
    print("Training Decision Tree model...")
    grid_search.fit(X_train_scaled, y_train)
    
    # Make predictions
    predictions = grid_search.predict(X_test_scaled)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # Print model details
    print("\nDecision Tree Details:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    print(f"Feature importances:")
    importances = grid_search.best_estimator_.feature_importances_
    for i, importance in enumerate(importances):
        if importance > 0.05:  # Display only features with importance above 5%
            print(f"Feature {i}: {importance:.3f}")
    
    return rmse, r2

if __name__ == '__main__':
    from data_preprocessing import preprocess_data
    features, target = preprocess_data('Housing.csv')
    print(decision_tree_model(features, target))