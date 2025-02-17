import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import hashlib

def preprocess_data(df):
    # Clean Size column (remove '�' character if present)
    df['Size'] = df['Size'].str.replace('�', '').astype(float)
    
    # Convert image URLs to hash values for classification
    df['Architecture'] = df['Architecture'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    
    # Initialize label encoders
    le_area = LabelEncoder()
    le_arch = LabelEncoder()
    
    # Encode categorical variables
    df['Area_encoded'] = le_area.fit_transform(df['Area'])
    df['Architecture_encoded'] = le_arch.fit_transform(df['Architecture'])
    
    return df, le_area, le_arch

def train_model(data_path):
    # Read the CSV file
    df = pd.read_csv(data_path, encoding='latin-1')
    
    # Preprocess the data
    df, area_encoder, arch_encoder = preprocess_data(df)
    
    # Select features and target
    X = df[['Size', 'Area_encoded']]
    y = df['Architecture_encoded']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Calculate accuracy
    accuracy = clf.score(X_test, y_test) * 100
    print(f"Model Accuracy: {accuracy:.2f}%")
    
    # Save model and encoders
    model_data = {
        'model': clf,
        'area_encoder': area_encoder,
        'architecture_encoder': arch_encoder,
        'valid_areas': list(df['Area'].unique()),
        'architecture_urls': dict(zip(df['Architecture_encoded'], df['Architecture']))
    }
    
    with open('property_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

if __name__ == '__main__':
    train_model('property_data.csv')