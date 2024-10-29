import joblib
import torch
import pandas as pd
from Accuracy_calculations.accuracy_checker_updated import calculate_accuracy_and_mae
from Accuracy_calculations.similarity_finder import find_closest_match
from distances_calculations.accuracy import extract_accuracy_distances
from angle_calculations.extract_angles import extract_angles
from angle_calculations.bat_angle import calculate_bat_angle
from bat.batgap import calculate_bat_gap
import cv2
from PIL import Image
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), 'best.pt')

# Load the classification model
clf = joblib.load(r'D:/Research/Cric-Boost/models/batsmen/New-trainedModel/random_forest_classification.pkl')

# Load the MinMaxScaler
scaler = joblib.load(r'D:/Research/Cric-Boost/models/batsmen/New_scalers/min_max_scaler.pkl')

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

def run_bat_detection(image_npa):
    # Load the YOLOv5 model        
    image_np = Image.open(image_npa)
    # image_resize_np = image_np.resize((640, 640))
    image_np = np.array(image_np)

    # Convert image to the format expected by YOLOv5
    results = model(image_np)

    # Print detected objects
    # print(results.pandas().xyxy[0])

    df = results.pandas().xyxy[0]
    cricket_bats = df[df['name'] == 'cricket-bats']
    if not cricket_bats.empty:
        # Get the cricket bat with the highest confidence
        highest_confidence_bat = cricket_bats.loc[cricket_bats['confidence'].idxmax()]

        bat_bbox = [
            highest_confidence_bat['xmin'], 
            highest_confidence_bat['ymin'], 
            highest_confidence_bat['xmax'], 
            highest_confidence_bat['ymax']
        ]
    else:
        return None
        
        # Print information about the highest-confidence cricket bat
        # print(highest_confidence_bat)
            
        # results.show()
    
    return highest_confidence_bat, bat_bbox

def predict(features, image_np, batsman_type, bat_bbox):
    try:
        # Convert the dictionary into a DataFrame with a single row
        df = pd.DataFrame([features])
        output_error = {}
        # Scale the features using the loaded MinMaxScaler
        scaled_features = scaler.transform(df)

        # Predict using the classifier with scaled features
        predicted_labels = clf.predict(scaled_features)
        confidence_levels = clf.predict_proba(scaled_features)

        # Get the confidence level for the predicted class
        predicted_class_confidence = max(confidence_levels[0])
        bat_angle = calculate_bat_angle(image_np, bat_bbox, predicted_labels[0], batsman_type)
        bat_gap = calculate_bat_gap(image_np, bat_bbox, predicted_labels[0], batsman_type)
        accuracy_distances = extract_accuracy_distances(image_np, batsman_type)
        closet_matches = find_closest_match(accuracy_distances, predicted_labels[0])

        angles = extract_angles(image_np, batsman_type)
        result = calculate_accuracy_and_mae(predicted_labels[0], angles, closet_matches, batsman_type, bat_angle, image_np, bat_gap)

        accuracy = result['Accuracy']
        rectifications = result['Rectification Messages']
        correct_angles = result['Correct Angles']
        bat_angle_info = result['Bat Angle Info']
        bat_gap_info = bool(result['Bat Gap'])
        unique_factors = result['Unique factors']

        if predicted_class_confidence < 0.5:
            output_error = 'The pose is not recognizable'
            output_data = {
                'response': output_error,
                'Stroke': predicted_labels[0], 
                # 'Confidence Levels': {shot_type: confidence for shot_type, confidence in zip(clf.classes_, confidence_levels[0])},
                'Highest Confidence Level': predicted_class_confidence
            }
        else:
            output_data = {
                'accuracy':f'{accuracy}%',
                'correct_angles':correct_angles,
                'rectifications':rectifications,
                'Stroke': predicted_labels[0],
                'bat_angle_info': bat_angle_info,
                'bat_gap_info': bat_gap_info,
                'unique_factors': unique_factors,
                # 'Confidence Levels': {shot_type: confidence for shot_type, confidence in zip(clf.classes_, confidence_levels[0])},
                'Highest Confidence Level': predicted_class_confidence
            }

            output_data= convert_numpy_types(output_data)

    except Exception as e:
        output_data = {
            'error': str(e)
        }

    return output_data

#REctifications with out classifications
def rectification_process(image_np, batsman_type, stroke, bat_bbox):
    try:
        #body landmark distances 
        accuracy_distances = extract_accuracy_distances(image_np, batsman_type)
        closet_matches = find_closest_match(accuracy_distances, stroke)
        
        angles = extract_angles(image_np, batsman_type)
        bat_angle = calculate_bat_angle(image_np, bat_bbox, stroke, batsman_type)
        bat_gap = calculate_bat_gap(image_np, bat_bbox, stroke, batsman_type)
        result = calculate_accuracy_and_mae(stroke, angles, closet_matches, batsman_type, bat_angle, image_np, bat_gap)

        #retrieveing the output
        accuracy = result['Accuracy']
        rectifications = result['Rectification Messages']
        correct_angles = result['Correct Angles']
        bat_angle_info = result['Bat Angle Info']
        bat_gap_info = bool(result['Bat Gap'])
        unique_factors = result['Unique factors']

        output_data = {
                'accuracy':f'{accuracy}%',
                'correct_angles':correct_angles,
                'rectifications':rectifications,
                'Stroke': stroke,
                'bat_angle_info': bat_angle_info,
                'bat_gap_info': bat_gap_info,
                'unique_factors': unique_factors,
            }
        
        output_data= convert_numpy_types(output_data)

    except Exception as e:
        output_data = {
            'error': str(e)
        }

    return output_data

def convert_numpy_types(data):
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.bool_):  # Check for numpy boolean
        return bool(data)  # Convert to native Python boolean
    return data