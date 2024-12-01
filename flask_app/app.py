import os
import sys
import numpy as np
import json
import pickle


from keras.models import load_model
from werkzeug.utils import secure_filename
from utils.utils import process_predictions
from feature_extraction.feature_extractor import FeatureExtractor
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'ogg'}

# Create Flask App
app = Flask(__name__)

# Limit content size
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_sample(filepath, latitude, longitude):
    feature_extractor = FeatureExtractor()
    return feature_extractor.process_sample(filepath, latitude, longitude)


# Upload files function
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Get latitude and longitude from the form
            latitude = request.form.get('latitude')
            longitude = request.form.get('longitude')

            if not latitude or not longitude:
                flash('Latitude and longitude must be provided.')
                return redirect(request.url)

            try:
                latitude = float(latitude)
                longitude = float(longitude)
            except ValueError:
                flash('Latitude and longitude must be valid numbers.')
                return redirect(request.url)

            # Save the file
            if file:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(filename)
                # Redirect to results page with file and location info
                return redirect(url_for('classify_and_show_results', filename=filename, latitude=latitude, longitude=longitude))
            else:
                return "No file uploaded."   
    return render_template("index.html")

# Classify and show results
@app.route('/results', methods=['GET'])
def classify_and_show_results():
    filename = request.args.get('filename')
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')

    try:
        latitude = float(latitude)  # Convert to float
        longitude = float(longitude)  # Convert to float
    except ValueError:
        return "Latitude and Longitude must be valid numbers.", 400

    try:
        # Compute audio signal features
        features = process_sample(filename, latitude, longitude)
    
    except Exception as e:
        app.logger.error(f"Error processing sample: {e}")
        return "Error processing sample.", 500

    features = np.expand_dims(features, 0)
    
    # Load model and perform inference
    model_path = 'models/XGBoost_Order.pkl'  
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Perform inference and get class probabilities
    predictions = model.predict(features)[0]
    predictions_probability = model.predict_proba(features)[0]
    
    # Load class labels from JSON
    with open('config_files/classes.json', 'r') as file:
        class_dictionary = json.load(file)
        # Sort classes by keys
        prediction_classes = np.array([class_dictionary[key] for key in sorted(class_dictionary.keys())])
    
    # Dynamically determine the number of available predictions
    num_predictions = min(len(predictions_probability), 3)
    
    # Process only the available predictions
    top_predictions_indices = np.argsort(predictions_probability)[::-1]  # Sort probabilities in descending order
    predictions_to_render = {
        prediction_classes[top_predictions_indices[i]]: "{}%".format(round(predictions_probability[top_predictions_indices[i]] * 100, 3))
        for i in range(num_predictions)
    }
    
    # Delete uploaded file
    os.remove(filename)
    
    # Render results
    return render_template("results.html", filename=filename, latitude=latitude, longitude=longitude, predictions_to_render=predictions_to_render)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

