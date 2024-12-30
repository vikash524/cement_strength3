import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request
import logging

# Application logging
logging.basicConfig(filename='deployment_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')  # Configuring logging operations

app = Flask(__name__)

# Load the XGBoost model from the .json file (XGBoost specific method)
model = xgb.Booster()
model.load_model(r'D:\cement_strenght3\Concrete-Compressive-Strength-Prediction\XGBoost_Regressor_model.json')

@app.route('/')
def home():
    return render_template('index.html')  # Renders the home page with the input form

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    For rendering results on HTML GUI and making predictions
    """
    if request.method == "POST":
        try:
            # Get inputs from the form and ensure they are valid numbers
            age = request.form.get('age')
            cement = request.form.get('cement')
            water = request.form.get('water')
            fa = request.form.get('fa')
            sp = request.form.get('sp')
            bfs = request.form.get('bfs')

            # Check if any field is empty
            if not age or not cement or not water or not fa or not sp or not bfs:
                return render_template('index.html', prediction_text="All fields are required. Please enter valid inputs.")

            # Attempt to convert the inputs to floats
            try:
                f_list = [
                    float(age), float(cement), float(water), 
                    float(fa), float(sp), float(bfs)
                ]
            except ValueError as e:
                return render_template('index.html', prediction_text="Invalid input values. Please enter numeric values.")

            # Logging inputs for debugging
            logging.info(f"Inputs received - Age: {f_list[0]}, Cement: {f_list[1]}, Water: {f_list[2]}, "
                         f"Fly Ash: {f_list[3]}, Superplasticizer: {f_list[4]}, Blast Furnace Slag: {f_list[5]}")

            # Prepare the input features for prediction (reshape to 2D array for XGBoost)
            final_features = np.array(f_list).reshape(1, -1)

            # Convert features to DMatrix for prediction (XGBoost format)
            dmatrix = xgb.DMatrix(final_features)

            # Predict using the model
            prediction = model.predict(dmatrix)
            result = "%.2f" % round(prediction[0], 2)  # Format result to two decimal places

            # Log the prediction
            logging.info(f"The predicted concrete compressive strength is {result} MPa")

            return render_template('index.html', prediction_text=f"The Concrete Compressive Strength is {result} MPa")

        except Exception as e:
            # General error handling
            logging.error(f"Error during prediction: {e}")
            return render_template('index.html', prediction_text="There was an error in prediction. Please try again.")

    # Handle GET request (for initial form rendering)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)


 