import os
import pandas as pd
from flask import Flask, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

class PropertyPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.features = ['Size', 'Area']
        
    def preprocess_data(self, df):
        # Create copy to avoid modifying original data
        df_processed = df.copy()
        
        # Convert Size to numeric, removing any non-numeric characters
        df_processed['Size'] = pd.to_numeric(df_processed['Size'].str.replace('[^0-9.]', '', regex=True))
        
        # Encode categorical variables
        self.label_encoders['Area'] = LabelEncoder()
        df_processed['Area'] = self.label_encoders['Area'].fit_transform(df_processed['Area'])
        
        # Create label encoder for target variable
        self.label_encoders['Architecture'] = LabelEncoder()
        df_processed['Architecture'] = self.label_encoders['Architecture'].fit_transform(df_processed['Architecture'])
        
        return df_processed
    
    def train(self, data_path):
        # Read and preprocess data
        df = pd.read_csv(data_path)
        df_processed = self.preprocess_data(df)
        
        # Prepare features and target
        X = df_processed[self.features]
        y = df_processed['Architecture']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Calculate and return accuracy
        accuracy = self.model.score(X_test, y_test)
        return accuracy
    
    def predict(self, size, area):
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Encode area using fitted encoder
        area_encoded = self.label_encoders['Area'].transform([area])[0]
        
        # Make prediction
        prediction = self.model.predict([[size, area_encoded]])[0]
        
        # Decode prediction
        architecture_type = self.label_encoders['Architecture'].inverse_transform([prediction])[0]
        
        return architecture_type
    
    def save_model(self, path='model.pkl'):
        if self.model is None:
            raise ValueError("No model to save!")
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path='model.pkl'):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']

# Initialize predictor
predictor = PropertyPredictor()

@app.route('/train', methods=['GET'])
def train_model():
    try:
        # Load CSV file from root directory
        data_path = 'property_data.csv'
        
        if not os.path.exists(data_path):
            return jsonify({'error': f'File {data_path} not found in root directory'}), 400
            
        # Train model
        accuracy = predictor.train(data_path)
        
        # Save model
        predictor.save_model()
        
        # Return the URL of the trained model
        return jsonify({
            'message': 'Model trained successfully',
            'accuracy': accuracy,
            'model_url': '/model'  # Example URL where you can access the model
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'size' not in data or 'area' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400
            
        # Load model if not already loaded
        if predictor.model is None:
            predictor.load_model()
            
        # Make prediction
        size = float(data['size'])
        area = data['area']
        
        prediction = predictor.predict(size, area)
        
        return jsonify({
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
