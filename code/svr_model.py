from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def svr_model(X, y):
    # חלוקת הנתונים
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # הגדרת פרמטרים לחיפוש
    param_grid = {
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    # חיפוש הפרמטרים הטובים ביותר
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # שימוש במודל הטוב ביותר
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    
    # חישוב מדדי ביצוע
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # הדפסת מידע נוסף
    print("\nSVR Details:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    return rmse, r2

if __name__ == '__main__':
    from data_preprocessing import preprocess_data
    features, target = preprocess_data('Housing.csv')
    print(svr_model(features, target))