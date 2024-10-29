from flask import Flask, request, jsonify  # type: ignore
import numpy as np
import cv2
from flask_cors import CORS
import traceback
from PIL import Image

from distances_calculations.classification import extract_distances
from util.prediction import predict, rectification_process, run_bat_detection

app = Flask(__name__)
CORS(app)

def process_image_data(image_file, batsman_type, has_classification, stroke):
    try:

        if image_file is None:
            raise ValueError("Could not read the image. Please check the file format.")

        image_bytes = image_file.read()
        image_file_bytes = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(image_file_bytes, cv2.IMREAD_COLOR)
        # Run bat detection
        highest_confidence_bat, bat_bbox = run_bat_detection(image_file)

        if has_classification == 'yes':
            distances = extract_distances(image_np, batsman_type)

            # Check if distances were extracted successfully
            if distances is None:
                return None, "No poses detected in the image"
            
            # Classification, accuracy calculations, and provide rectifications
            predicted_stroke_data = predict(distances, image_np, batsman_type, bat_bbox)
        else:
            predicted_stroke_data = rectification_process(image_np, batsman_type, stroke, bat_bbox)

        return predicted_stroke_data
    except Exception as e:
        print("Exception occurred:", e)
        print(traceback.format_exc())  # Print detailed traceback to the console
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    if 'stroke' in request.form:
        stroke = request.form['stroke']
    else:
        stroke = None  # Set to None if not provided

    if 'classify' not in request.form:
        return jsonify({"error": "Enter the classification status"}), 400

    if 'type' not in request.form:
        return jsonify({"error": "Enter the batsman type"}), 400

    batsman_type = request.form['type']
    has_classification = request.form['classify']

    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        # Process the image directly from the file
        output_data = process_image_data(image_file, batsman_type, has_classification, stroke)
        print(output_data)
        # Return the result
        return jsonify(output_data), 200

    except Exception as e:
        print("Exception occurred:", e)
        print(traceback.format_exc())  # Print detailed traceback to the console
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5050)
