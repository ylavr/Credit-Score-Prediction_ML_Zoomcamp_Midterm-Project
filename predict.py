from flask import Flask, request, jsonify
import requests
import pickle
import pandas as pd
import numpy as np
import json
from colorama import Fore, Style

# # Initialize Flask app
app = Flask('predict_app')

# Define the helper function
def predict_single(input_data, dict_vectorizer, model, label_encoder):
    """
    Transforms input data using DictVectorizer, makes predictions with the model,
    and decodes the predicted class.
    """
    try:
                     
        test = json.loads(input_data)
        input_features = dict_vectorizer.transform([test])
        # input_features = dict_vectorizer.transform([input_data])
        print(f"Step 2: Transformed features:, input_features\n")


        # Make predictions
        predicted_class_index = model.predict(input_features)
        predicted_probabilities = model.predict_proba(input_features)
        print(f"Step 3: Predictions made successfully.\n")

        # Decode the predicted class
        predicted_class = label_encoder.inverse_transform(predicted_class_index)[0]
        #print(f"Predicted class: {predicted_class}")
        print(f"{Fore.RED}{Style.BRIGHT}Predicted class: {predicted_class}{Style.RESET_ALL}")

        # Return results
        return {
            "predicted_class": predicted_class,
            "predicted_probabilities": dict(
                zip(label_encoder.classes_, predicted_probabilities[0].tolist())  # Map probabilities to class names
            )
        }
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")
    
@app.route('/', methods=['GET'])
def Welcome_message():
    return jsonify({"message": "Welcome to the prediction service!"})


# Define predict route
@app.route("/predict", methods=["POST"])
def predict():
    """
    Flask route for handling predictions. Receives JSON input,
    uses the helper function to generate predictions, and returns the result.
    """
    try:
        print(f"\n*************Prediction started********************")
        # Load the model, DictVectorizer, and LabelEncoder
        
        with open("final_model.pkl", "rb") as f_model:
            model = pickle.load(f_model)
        with open("dict_vectorizer.pkl", "rb") as f_dv:
            dict_vectorizer = pickle.load(f_dv)
        with open("label_encoder.pkl", "rb") as f_le:
            label_encoder = pickle.load(f_le)
        print(f"\nLoaded model, DictVectorizer, and LabelEncoder.\n")

        # Parse input JSON
        input_data = request.get_json()
        print("Received input:", input_data)
        if not input_data:
            return jsonify({"error": "No input data provided!"}), 400
        print(f"Step 1: Received input data: {input_data}\n")

        # Use helper function for prediction
        response = predict_single(input_data, dict_vectorizer, model, label_encoder)

        print(f"\n*************Prediction finished********************\n")
        print(response)
        return jsonify(response)

    except ValueError as ve:
        print(f"Prediction error: {str(ve)}")
        return jsonify({"error": f"Prediction failed: {str(ve)}"}), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    