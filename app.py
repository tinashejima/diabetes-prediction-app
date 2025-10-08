
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import shap

app = Flask(__name__)

# Load the model
model = pickle.load(open('rf_model.pkl', 'rb'))

# Initialize SHAP explainer (do this once at startup)
explainer = None

def initialize_explainer():
    """Initialize SHAP explainer with background data"""
    global explainer
    try:
        # Create a small background dataset with typical values
        background_data = pd.DataFrame({
            'age': [40, 50, 60, 30, 70],
            'hypertension': [0, 1, 0, 0, 1],
            'heart_disease': [0, 0, 1, 0, 1],
            'bmi': [25, 30, 28, 22, 32],
            'HbA1c_level': [5.5, 6.0, 5.8, 5.2, 6.5],
            'blood_glucose_level': [100, 120, 110, 90, 140],
            'smoking_history_No Info': [0, 0, 0, 1, 0],
            'smoking_history_current': [0, 1, 0, 0, 0],
            'smoking_history_ever': [0, 0, 0, 0, 0],
            'smoking_history_former': [0, 0, 1, 0, 0],
            'smoking_history_never': [1, 0, 0, 0, 1],
            'smoking_history_not current': [0, 0, 0, 0, 0],
            'gender_Female': [1, 0, 1, 0, 1],
            'gender_Male': [0, 1, 0, 1, 0],
            'gender_Other': [0, 0, 0, 0, 0]
        })
        
        # Use TreeExplainer with proper model_output parameter
        explainer = shap.TreeExplainer(model, background_data)
        print("SHAP explainer initialized successfully")
    except Exception as e:
        print(f"Error initializing SHAP explainer: {str(e)}")
        raise

def encode_features(data):
    """Encode categorical features to match the model's expected format"""
    encoded_data = {
        'age': float(data['age']),
        'hypertension': int(data['hypertension']),
        'heart_disease': int(data['heart_disease']),
        'bmi': float(data['bmi']),
        'HbA1c_level': float(data['HbA1c_level']),
        'blood_glucose_level': float(data['blood_glucose_level']),
        'smoking_history_No Info': 0,
        'smoking_history_current': 0,
        'smoking_history_ever': 0,
        'smoking_history_former': 0,
        'smoking_history_never': 0,
        'smoking_history_not current': 0,
        'gender_Female': 0,
        'gender_Male': 0,
        'gender_Other': 0
    }
    
    # Set the appropriate smoking history column to 1
    smoking_col = f"smoking_history_{data['smoking_history']}"
    if smoking_col in encoded_data:
        encoded_data[smoking_col] = 1
    
    # Set the appropriate gender column to 1
    gender_col = f"gender_{data['gender']}"
    if gender_col in encoded_data:
        encoded_data[gender_col] = 1
    
    return encoded_data

def get_feature_importance_explanation(df, shap_values):
    """
    Get top factors influencing the prediction with readable names using SHAP values
    """
    # Handle SHAP values based on output format
    print(f"SHAP values type: {type(shap_values)}")
    print(f"SHAP values shape/length: {shap_values.shape if hasattr(shap_values, 'shape') else len(shap_values)}")
    
    # For TreeExplainer with probability output, we get an array directly
    if isinstance(shap_values, np.ndarray):
        # Check if it's 2D (single prediction with multiple features)
        if len(shap_values.shape) == 2:
            shap_vals = shap_values[0]  # Get first row
        else:
            shap_vals = shap_values
    elif isinstance(shap_values, list):
        # If it's a list (for binary classification without probability output)
        if len(shap_values) == 2:
            # Get SHAP values for positive class (diabetes = 1)
            if isinstance(shap_values[1], np.ndarray):
                shap_vals = shap_values[1][0] if len(shap_values[1].shape) > 1 else shap_values[1]
            else:
                shap_vals = shap_values[1]
        else:
            shap_vals = shap_values[0][0] if len(shap_values[0].shape) > 1 else shap_values[0]
    else:
        # Fallback: convert to numpy array
        shap_vals = np.array(shap_values).flatten()
    
    print(f"Processed SHAP values shape: {shap_vals.shape}")
    print(f"Number of features in df: {len(df.columns)}")
    
    # Create a mapping of encoded features to readable names
    feature_names = {
        'age': 'Age',
        'hypertension': 'Hypertension',
        'heart_disease': 'Heart Disease',
        'bmi': 'BMI',
        'HbA1c_level': 'HbA1c Level',
        'blood_glucose_level': 'Blood Glucose Level',
        'smoking_history_No Info': 'Smoking History: No Info',
        'smoking_history_current': 'Smoking History: Current',
        'smoking_history_ever': 'Smoking History: Ever',
        'smoking_history_former': 'Smoking History: Former',
        'smoking_history_never': 'Smoking History: Never',
        'smoking_history_not current': 'Smoking History: Not Current',
        'gender_Female': 'Gender: Female',
        'gender_Male': 'Gender: Male',
        'gender_Other': 'Gender: Other'
    }
    
    # Get feature names and their SHAP values
    features = df.columns.tolist()
    feature_values = df.iloc[0].values
    
    # Combine features with their SHAP values and actual values
    importance_data = []
    for i, feature in enumerate(features):
        shap_value = shap_vals[i] if i < len(shap_vals) else 0
        
        importance_data.append({
            'feature': feature_names.get(feature, feature),
            'shap_value': abs(shap_value),
            'actual_shap': shap_value,
            'value': feature_values[i],
            'impact': 'Increases' if shap_value > 0 else 'Decreases'
        })
    
    # Sort by absolute SHAP value
    importance_data.sort(key=lambda x: x['shap_value'], reverse=True)
    
    # Get top 5 factors
    top_factors = []
    for item in importance_data[:5]:
        if item['shap_value'] > 0.001:  # Only include meaningful contributions
            # Format the value display
            if item['value'] in [0, 1]:
                value_display = 'Yes' if item['value'] == 1 else 'No'
            else:
                value_display = f"{item['value']:.2f}"
            
            top_factors.append({
                'feature': item['feature'],
                'value': value_display,
                'impact': item['impact'],
                'importance': round(item['shap_value'] * 100, 2)
            })
    
    return top_factors

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Initialize explainer if not already done
        if explainer is None:
            initialize_explainer()
        
        # Get form data
        data = {
            'age': request.form['age'],
            'gender': request.form['gender'],
            'bmi': request.form['bmi'],
            'HbA1c_level': request.form['HbA1c_level'],
            'blood_glucose_level': request.form['blood_glucose_level'],
            'smoking_history': request.form['smoking_history'],
            'hypertension': request.form['hypertension'],
            'heart_disease': request.form['heart_disease']
        }
        
        print("Received data:", data)
        
        # Encode features
        encoded_data = encode_features(data)
        
        # Convert to DataFrame with correct column order
        feature_columns = [
            'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
            'blood_glucose_level', 'smoking_history_No Info', 'smoking_history_current',
            'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
            'smoking_history_not current', 'gender_Female', 'gender_Male', 'gender_Other'
        ]
        
        df = pd.DataFrame([encoded_data], columns=feature_columns)
        
        print("DataFrame for prediction:")
        print(df)
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] * 100
        
        # Calculate SHAP values for interpretation
        try:
            # Get SHAP values - for binary classification, this returns a list of 2 arrays
            shap_values = explainer.shap_values(df)
            
            # Debug: Print SHAP values structure
            print(f"SHAP values type: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"SHAP values is a list with {len(shap_values)} elements")
                for i, sv in enumerate(shap_values):
                    print(f"  Element {i} shape: {sv.shape if hasattr(sv, 'shape') else 'N/A'}")
            else:
                print(f"SHAP values shape: {shap_values.shape}")
            
            top_factors = get_feature_importance_explanation(df, shap_values)
            print(f"Top factors calculated successfully: {len(top_factors)} factors")
        except Exception as shap_error:
            print(f"SHAP calculation error: {str(shap_error)}")
            import traceback
            traceback.print_exc()
            # Fallback to empty factors if SHAP fails
            top_factors = []
        
        print(f"Prediction: {prediction}, Probability: {probability}")
        
        # Calculate risk level
        if probability < 20:
            risk_level = "Low"
            risk_color = "#28a745"
        elif probability < 50:
            risk_level = "Moderate"
            risk_color = "#ffc107"
        else:
            risk_level = "High"
            risk_color = "#dc3545"
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': round(probability, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'top_factors': top_factors
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)