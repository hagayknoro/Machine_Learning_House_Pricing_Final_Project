from data_preprocessing import preprocess_data
from linear_regression_model import linear_regression_model
from decision_tree_model import decision_tree_model
from svr_model import svr_model
from knn_model import knn_model
import numpy as np
import time

if __name__ == '__main__':
    # Data processing without PCA
    features, target = preprocess_data('Housing.csv', n_features=10, use_pca=False)
    
    print(f"\nTotal number of samples: {features.shape[0]}")
    print(f"Number of features: {features.shape[1]}")
    
    # Train models with timing
    print("\nTraining models...")
    
    start_time = time.time()
    lr_rmse, lr_r2 = linear_regression_model(features, target)
    print(f"Linear Regression training time: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    dt_rmse, dt_r2 = decision_tree_model(features, target)
    print(f"Decision Tree training time: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    svr_rmse, svr_r2 = svr_model(features, target)
    print(f"SVR training time: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    knn_rmse, knn_r2 = knn_model(features, target)
    print(f"KNN training time: {time.time() - start_time:.2f} seconds")
    
    # Model comparison
    print("\nModel Comparison:")
    print(f"Linear Regression: RMSE={np.exp(lr_rmse):.0f}, R²={lr_r2:.3f}")
    print(f"Decision Tree: RMSE={np.exp(dt_rmse):.0f}, R²={dt_r2:.3f}")
    print(f"SVR: RMSE={np.exp(svr_rmse):.0f}, R²={svr_r2:.3f}")
    print(f"KNN: RMSE={np.exp(knn_rmse):.0f}, R²={knn_r2:.3f}")