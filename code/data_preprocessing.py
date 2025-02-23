import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.linear_model import Lasso

def preprocess_data(file_name, n_features='auto', use_pca=False):
    # Build the correct path to the data file
    data_dir = 'data'
    file_path = os.path.join(data_dir, file_name)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_name} was not found in {data_dir} directory. "
                              f"Please make sure to place your data file in the {data_dir} folder.")
    
    # Read the data
    data = pd.read_csv(file_path)
    
    # Print data info for debugging
    print("Available columns:", data.columns.tolist())
    
    # Separate target variable
    if 'price' not in data.columns:
        raise ValueError("Column 'price' not found in the dataset. Available columns are: "
                        f"{', '.join(data.columns.tolist())}")
    
    # טיפול בערכים חריגים
    def remove_outliers(df, columns, n_std=3):
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[np.abs(df[col] - mean) <= (n_std * std)]
        return df
    
    # הסרת ערכים חריגים ממשתנים מספריים
    numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    data = remove_outliers(data, numeric_features)
    
    # יצירת תכונות מורכבות יותר
    # תכונות קיימות
    data['rooms_per_floor'] = data['bedrooms'] / data['stories']
    data['total_rooms'] = data['bedrooms'] + data['bathrooms']
    data['area_per_room'] = data['area'] / data['total_rooms']
    data['area_per_floor'] = data['area'] / data['stories']
    
    # טרנספורמציות לוגריתמיות
    data['log_area'] = np.log1p(data['area'])
    data['log_price'] = np.log1p(data['price'])  # המחיר גם
    
    # תכונות ריבועיות
    data['area_squared'] = data['area'] ** 2
    data['stories_squared'] = data['stories'] ** 2
    
    # אינטראקציות מורכבות
    data['area_rooms_ratio'] = data['area'] / (data['bedrooms'] + data['bathrooms'])
    data['luxury_score'] = ((data['airconditioning'] == 'yes').astype(int) + 
                           (data['hotwaterheating'] == 'yes').astype(int) +
                           (data['guestroom'] == 'yes').astype(int) +
                           (data['basement'] == 'yes').astype(int))
    
    # אינטראקציות עם משתנים קטגוריים
    data['premium_area'] = data['area'] * (data['prefarea'] == 'yes').astype(int)
    
    # הוספת אינטראקציות עם log_area
    data['log_area_rooms'] = data['log_area'] * data['total_rooms']
    data['log_area_stories'] = data['log_area'] * data['stories']
    
    # הוספת תכונות פשוטות יותר
    data['area_per_bathroom'] = data['area'] / data['bathrooms']
    data['rooms_ratio'] = data['bedrooms'] / data['bathrooms']
    
    # הפרדת משתנה המטרה (עכשיו לוגריתמי)
    target = data['log_price']
    features = data.drop(['price', 'log_price'], axis=1)
    
    # זיהוי סוגי עמודות
    numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = features.select_dtypes(include=['object']).columns
    
    # יצירת Pipeline לטיפול בנתונים
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    # עיבוד ראשוני של הנתונים
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # עיבוד הנתונים
    features_processed = preprocessor.fit_transform(features)
    
    # Get feature names
    numeric_features_list = list(numeric_features)
    categorical_features_list = []
    if len(categorical_features) > 0:
        categorical_features_list = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    
    # Create DataFrame with processed features
    feature_names = numeric_features_list + list(categorical_features_list)
    features_processed_df = pd.DataFrame(features_processed, columns=feature_names)
    
    # בחירת תכונות עם מספר דינמי
    if n_features == 'auto':
        n_features = min(8, features_processed_df.shape[1])
    
    selector = SelectFromModel(
        estimator=Lasso(alpha=0.01),
        max_features=10
    )
    features_selected = selector.fit_transform(features_processed_df, target)
    
    # הדפסת התכונות שנבחרו
    selected_features = features_processed_df.columns[selector.get_support()].tolist()
    print(f"\nSelected features: {selected_features}")
    
    # במקום להדפיס scores, נדפיס את המקדמים של ה-Lasso
    if hasattr(selector.estimator_, 'coef_'):
        feature_importance = np.abs(selector.estimator_.coef_)
        print(f"Feature importance (Lasso coefficients): {feature_importance[selector.get_support()]}")
    
    # PCA רק אם מבקשים
    if use_pca:
        pca = PCA(n_components=0.98)  # שומרים על 98% מהשונות
        features_final = pca.fit_transform(features_selected)
        print(f"Variance explained by PCA: {sum(pca.explained_variance_ratio_):.3f}")
    else:
        features_final = features_selected
    
    return features_final, target

if __name__ == '__main__':
    features, target = preprocess_data('Housing.csv')
    print(f"Processed features shape: {features.shape}")
    if target is not None:
        print(f"Target shape: {target.shape}")