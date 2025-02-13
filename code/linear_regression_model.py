from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def linear_regression_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the models
    lasso_model = LassoCV(cv=5, random_state=42)
    ridge_model = RidgeCV(cv=5)
    
    lasso_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    
    # Make predictions
    lasso_pred = lasso_model.predict(X_test)
    ridge_pred = ridge_model.predict(X_test)
    
    lasso_r2 = r2_score(y_test, lasso_pred)
    ridge_r2 = r2_score(y_test, ridge_pred)
    
    # Select the better model
    if lasso_r2 > ridge_r2:
        predictions = lasso_pred
        model = lasso_model
    else:
        predictions = ridge_pred
        model = ridge_model
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # Print additional information
    print("\nLinear Regression Details:")
    print(f"Selected model: {'Lasso' if lasso_r2 > ridge_r2 else 'Ridge'}")
    if isinstance(model, LassoCV):
        print(f"Number of features used: {np.sum(model.coef_ != 0)}")
    print(f"Cross-validation score: {np.mean(cross_val_score(model, X, y, cv=5)):.3f}")
    
    return rmse, r2

if __name__ == '__main__':
    from data_preprocessing import preprocess_data
    features, target = preprocess_data('Housing.csv')
    print(linear_regression_model(features, target))