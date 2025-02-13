from data_preprocessing import preprocess_data
from linear_regression_model import linear_regression_model
from decision_tree_model import decision_tree_model
from svr_model import svr_model
from knn_model import knn_model

if __name__ == '__main__':
    # Preprocess data
    features, target = preprocess_data('Housing.csv')

    # Run models
    lr_rmse, lr_r2 = linear_regression_model(features, target)
    dt_rmse, dt_r2 = decision_tree_model(features, target)
    svr_rmse, svr_r2 = svr_model(features, target)
    knn_rmse, knn_r2 = knn_model(features, target)

    # Compare models
    print(f"Linear Regression RMSE: {lr_rmse}, R²: {lr_r2}")
    print(f"Decision Tree RMSE: {dt_rmse}, R²: {dt_r2}")
    print(f"SVR RMSE: {svr_rmse}, R²: {svr_r2}")
    print(f"KNN RMSE: {knn_rmse}, R²: {knn_r2}")