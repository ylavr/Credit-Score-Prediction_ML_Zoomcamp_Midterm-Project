import requests
from flask import Flask, request, jsonify
import pandas as pd
import json
import colorama

#Please use it when test locally
url = "http://127.0.0.1:9696/predict"

# Load a test row from df_test.csv
df_test = pd.read_csv("df_test.csv")


while True:
    # Prompt the user for a row index
    print(f"Dataset preview: length of df is {len(df_test)} rows")
    print(df_test.head(10))  

    print("Before we pass the selected row to the model, we drop the target column 'credit_score'. This column is shown above just for comparison with the prediction result.")

    while True:
        try:
            row_index = int(input(f"\nEnter the index of the row you want to use for prediction: "))
            # row_index = 2
            if row_index < 0 or row_index >= len(df_test):
                print(f"Invalid index. Please enter a number between 0 and {len(df_test) - 1}.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    # Prepare the selected row for prediction
    test_row = df_test.drop(columns=['credit_score']).iloc[row_index].to_dict()  # Drop the target column
    input_data = json.dumps(test_row) 

    print(f"\nSelected row data (as JSON):")
    print(input_data)

    # Send the POST request to the /predict endpoint
    response = requests.post(url, json=input_data)

    # Print the server response
    if response.status_code == 200:
        print(f"\n{'='*30} Prediction Response {'='*30}\n")
        prediction = response.json()
        
        # Extract and print the predicted class
        predicted_class = prediction['predicted_class']
        print(f"Predicted Class: \033[1m\033[91m{predicted_class}\033[0m\n") 

        # Extract and format probabilities
        probabilities = prediction['predicted_probabilities']
        print("Predicted Probabilities:")
        for cls, prob in probabilities.items():
            print(f"  - {cls}: {round(prob * 100, 2)}%") 
        
        print(f"\n{'='*74}\n")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

    # Ask the user if they want to test another row
    continue_testing = input("Do you want to test another row? (yes/no): ").strip().lower()
    if continue_testing != 'yes':
        print("Exiting...")
        break
