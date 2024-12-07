from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model and preprocessing objects
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
activity_encoder = joblib.load('activity_encoder.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_weight_status():
    data = request.get_json()

    # Extract the data from the JSON request
    age = data['age']
    gender = data['gender']
    activity_level = data['activity_level']
    carbs = data['carbs']
    fats = data['fats']
    proteins = data['proteins']
    height = data['height']
    weight = data['weight']
    calorie_intake = data['calorie_intake']

    # Encode categorical variables
    gender_encoded = gender_encoder.transform([gender])[0]
    activity_encoded = activity_encoder.transform([activity_level])[0]

    # Prepare the numeric features (matching those used during training)
    numeric_features = np.array([[calorie_intake, fats, proteins, carbs, height, weight, age]])

    # Scale the numeric features
    numeric_features_scaled = scaler.transform(numeric_features)

    # Combine scaled numeric features with encoded categorical features
    input_data = np.hstack((numeric_features_scaled, [[gender_encoded, activity_encoded]]))

    # Predict using the trained model
    prediction = rf_model.predict(input_data)[0]

    # Map numerical predictions to meaningful labels
    weight_status_mapping = {0: "Lose Weight", 1: "Maintain Weight", 2: "Gain Weight"}
    result = weight_status_mapping[prediction]

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
