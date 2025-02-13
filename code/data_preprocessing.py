import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

def preprocess_data(file_name):
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
    
    # יצירת תכונות חדשות
    data['rooms_per_floor'] = data['bedrooms'] / data['stories']
    data['total_rooms'] = data['bedrooms'] + data['bathrooms']
    data['has_premium_features'] = ((data['airconditioning'] == 'yes') & 
                                  (data['hotwaterheating'] == 'yes')).astype(int)
    
    # הפרדת משתנה המטרה
    target = data['price']
    features = data.drop('price', axis=1)
    
    # זיהוי סוגי עמודות
    numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = features.select_dtypes(include=['object']).columns
    
    # יצירת Pipeline לטיפול בנתונים
    numeric_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())  # רק נרמול, בלי בחירת תכונות כאן
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
    
    # בחירת תכונות אחרי העיבוד הראשוני
    selector = SelectKBest(f_regression, k=5)
    features_selected = selector.fit_transform(features_processed_df, target)
    
    # Apply PCA if needed (optional)
    pca = PCA(n_components=0.95)  # Keep 95% of the variance
    principal_components = pca.fit_transform(features_selected)
    
    return principal_components, target

if __name__ == '__main__':
    features, target = preprocess_data('Housing.csv')
    print(f"Processed features shape: {features.shape}")
    if target is not None:
        print(f"Target shape: {target.shape}")