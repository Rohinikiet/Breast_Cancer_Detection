from flask import Flask, render_template, request, url_for # url_for might not be needed for this specific route anymore, but keep it for others
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import logging # Import logging

app = Flask(__name__)

# --- Setup Logging ---
# It's good practice to log errors, especially in production
logging.basicConfig(level=logging.INFO) # Log INFO level messages and above
# You might want to configure file logging for production

# Load CNN model and scaler
try:
    model = load_model('cancer_cnn_model.h5')
    scaler = joblib.load('scaler (1).pkl')
    app.logger.info("Model and scaler loaded successfully.") # Use app.logger
except Exception as e:
    app.logger.error(f"CRITICAL: Error loading model or scaler: {e}", exc_info=True) # Log the full traceback
    # Decide how to handle this - exiting might be okay for development,
    # but in production you might want the app to start but log the error
    print(f"CRITICAL: Error loading model or scaler: {e}") # Also print to console
    # exit() # Consider if you really want to exit

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict-form')
def predict_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        # Use request.form.get to avoid KeyError if a field is missing,
        # although the float conversion will still fail, which is caught below.
        size = float(request.form['size'])
        texture = float(request.form['texture'])
        perimeter = float(request.form['perimeter'])
        concavity = float(request.form['concavity'])
        features = [size, texture, perimeter, concavity]

        app.logger.info(f"Received features: {features}") # Log received data

        # Preprocess input
        input_scaled = scaler.transform([features])
        input_scaled = input_scaled.reshape(1, 4, 1)  # Shape for CNN model

        # Make prediction
        prediction_value = model.predict(input_scaled)[0][0] # Get the raw probability
        app.logger.info(f"Raw prediction value: {prediction_value}")

        image_url = "" # Initialize variable - use a more descriptive name
        result_text = "" # Initialize variable

        # Determine result text and image URL based on prediction threshold (0.5)
        if prediction_value < 0.5:
            result_text = "🔴 High Risk (Malignant)"
            # Assign the full URL directly
            image_url = "https://tse4.mm.bing.net/th?id=OIP.MLTTiuipPUc78GnQ-NaoRwAAAA&pid=Api&P=0&h=220"
        else:
            result_text = "🟢 Low Risk (Benign)"
            # Assign the full URL directly
            image_url = "https://gifdb.com/images/high/cute-happy-excited-cat-gx339xf5f5zewrju.gif"

        app.logger.info(f"Result: {result_text}, Image URL: {image_url}")

        # Pass the result text and the direct image URL to the template
        return render_template('result.html',
                               prediction=result_text,
                               result_image_url=image_url) # Pass the URL directly

    except KeyError as e:
         app.logger.warning(f"Prediction failed: Missing form field '{str(e)}'. Form data: {request.form}")
         return f"Prediction failed: Missing form field '{str(e)}'. Please go back and fill all fields."
    except ValueError as e:
        app.logger.warning(f"Prediction failed: Invalid input. Error: {str(e)}. Form data: {request.form}")
        return f"Prediction failed: Invalid input. Please ensure all inputs are numbers. Error: {str(e)}"
    except FileNotFoundError as e:
        app.logger.error(f"Prediction failed: Required file not found. {str(e)}", exc_info=True)
        return f"Prediction failed: Required file not found. {str(e)}"
    except Exception as e:
        # Log the detailed error for debugging on the server side
        app.logger.error(f"Prediction failed with unexpected error: {str(e)}", exc_info=True)
        # Show a generic error to the user
        return "Prediction failed due to an internal server error. Please try again later or contact support."


if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network (optional)
    app.run(debug=True, host='0.0.0.0') # debug=True provides more detailed errors