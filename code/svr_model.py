from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

def svr_model(X, y):
    start_time = time.time()
    print("Starting SVR training...")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split completed")
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data scaling completed")
    
    # Train a basic SVR model
    print("Training basic SVR model...")
    model = SVR(
        kernel='linear',
        C=10.0,
        epsilon=0.1,
        max_iter=10000,
        tol=1e-4
    )
    
    try:
        model.fit(X_train_scaled, y_train)
        print("Model training completed")
        
        predictions = model.predict(X_test_scaled)
        print("Predictions completed")
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        training_time = time.time() - start_time
        print(f"\nSVR Details:")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"R² score: {r2:.3f}")
        
        return rmse, r2
        
    except Exception as e:
        print(f"Error during SVR training: {str(e)}")
        return np.inf, -np.inf

if __name__ == '__main__':
    from data_preprocessing import preprocess_data
    features, target = preprocess_data('Housing.csv')
    print(svr_model(features, target))

param_grid = {
    'C': [1.0, 10.0, 100.0],
    'epsilon': [0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='r2', n_jobs=-1)