import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(raw_path="data/raw/Mall_Customers.csv", 
                    processed_path="data/processed/processed_data.csv"):
    """
    Loads raw data, selects numerical features, scales them,
    and saves the processed data for clustering.
    
    Args:
        raw_path (str): Path to the raw CSV file.
        processed_path (str): Path to save the processed CSV.
    
    Returns:
        processed_scaled (pd.DataFrame): Scaled numerical features.
    """
    
    # Ensure processed folder exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    # Load raw data
    raw_data = pd.read_csv(raw_path)
    
    # Check for missing values
    if raw_data.isnull().values.any():
        raise ValueError("Missing values detected in raw data. Handle them before preprocessing.")
    
    # Select numerical features
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    processed_data = raw_data[features]
    
    # Scale features
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(processed_data)
    processed_scaled = pd.DataFrame(scaled_array, columns=features)
    
    # Save processed data
    processed_scaled.to_csv(processed_path, index=False)
    
    return processed_scaled
