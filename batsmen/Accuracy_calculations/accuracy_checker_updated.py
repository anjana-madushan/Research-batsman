import pandas as pd
import numpy as np
from checker.f_defence import head_position_check_fd
from checker.f_drive import head_knee_toe_position_check_fdr
from checker.backfoot_shots import head_position_check_backfoot

def calculate_accuracy_and_mae(shot_type, input_angles, closet_matches, batsman_type, bat_angle, image_np, bat_gap):

    closet_matches = pd.DataFrame(closet_matches)

    mean_values = closet_matches.mean()
    std_values = closet_matches.std()

    angle_columns = closet_matches.columns.tolist()
    # Define deviation thresholds based on the standard deviations
    deviation_thresholds = {
        angle: (mean_values[angle], std_values[angle])
        for angle in angle_columns
        if angle in mean_values
    }

    # Check for empty deviation thresholds
    if not deviation_thresholds:
        raise ValueError("No valid deviation thresholds available")

    # Use the updated deviation thresholds in the calculation
    result = categorize_and_calculate_mae(input_angles, mean_values.to_dict(), deviation_thresholds, batsman_type, shot_type, bat_angle, image_np, bat_gap)

    return result

def categorize_and_calculate_mae(input_angles, reference_angles, deviation_thresholds, batsman_type, shot_type, batangle, image_np, bat_gap):
    absolute_deviations = []
    false_joints = {}
    incorrect_angles = {}
    rectification_needed = {}
    correctness = []
    correct_angles = []

    total_error = 0
    count = 0
    checker_result = None

    unique_rectifications = []

    bat_gap_checker = bat_gap <= get_average_bat_gap(shot_type)
        
    if shot_type == 'forward defence':
      checker_result = head_position_check_fd(image_np, batsman_type)
    elif shot_type == 'forward drive':
      checker_result = head_knee_toe_position_check_fdr(image_np, batsman_type)
    elif shot_type == 'backfoot defence' or shot_type == 'backfoot drive':
      checker_result = head_position_check_backfoot(image_np, batsman_type)
    unique = {
                'stroke': shot_type,
                'result':checker_result
            }

    unique_rectifications.append(unique)
    average_bat_angle, sd_bat_angle = get_average_bat_angle(shot_type)

    for angle, input_value in input_angles.items():
        correct_angle_name = correct_angle_name_converter(angle, batsman_type)
        if angle in reference_angles:
            reference_value = reference_angles[angle]
            accurate_threshold, minor_error_threshold = deviation_thresholds.get(angle, (float('inf'), float('inf')))

            error = abs(input_value - reference_value)

            if error < minor_error_threshold:
                correctness.append(100)
                correct_angles.append(correct_angle_name)
            elif error >= minor_error_threshold and error < 2*minor_error_threshold:
                absolute_deviations.append(error)
                false_joints[angle] = input_value
                rectification_needed[angle] = input_value
                total_error += error
                count += 1
            else:
                incorrect_angles[angle] = input_value
                rectification_needed[angle] = input_value
                correctness.append(0)

    bat_angle_error = abs(batangle - average_bat_angle) if average_bat_angle is not None else None
    bat_angle_within_range = bat_angle_error is not None and bat_angle_error <= sd_bat_angle  

    # Calculate bat angle error if available
    if average_bat_angle is not None:
        bat_angle_error = abs(batangle - average_bat_angle)
        absolute_deviations.append(bat_angle_error) 
        if bat_angle_error <= sd_bat_angle:  
            correctness.append(100)
            # correct_angles.append('bat_angle')
        else:
            correctness.append(0)

    overall_mae = sum(absolute_deviations) / len(absolute_deviations) if absolute_deviations else 0
    mae_percentage = round(100 - overall_mae)  #accurate percentage

    bat_angle_based = {
        'Bat Angle': batangle,
        'Bat Angle Error': bat_angle_error,
        'Bat Angle Within Range': bat_angle_within_range
    }
    
    rectifications = generate_rectification_messages(
        rectification_needed.items(),
        reference_angles,
        deviation_thresholds, 
        batsman_type
    )

    result = {
        'Accuracy': mae_percentage,
        'Rectification Messages': rectifications,
        'Correct Angles': correct_angles,
        'Bat Angle Info': bat_angle_based,
        'Bat Gap': bat_gap_checker,
        'Unique factors': unique_rectifications
    }

    return result

#generate rectification messages
def generate_rectification_messages(rectification_needed_items, reference_angles, deviation_thresholds, batsman_type):
    rectifications = []

    neighboring_joints = {
        'angle_right_elbow': ['RIGHT SHOULDER', 'RIGHT WRIST'],
        'angle_left_elbow': ['LEFT SHOULDER', 'LEFT WRIST'],
        'angle_right_shoulder': ['RIGHT HIP', 'RIGHT SHOULDER'],
        'angle_left_knee': ['LEFT HIP', 'LEFT ANKLE'],
        'angle_right_knee': ['RIGHT HIP', 'RIGHT ANKLE'],
        'angle_right_hip_knee': ['RIGHT KNEE', 'LEFT HIP'],
        'angle_left_hip_knee':['LEFT KNEE', 'RIGHT HIP'],
        'angle_right_hip_shoulder': ['RIGHT KNEE', 'RIGHT SHOULDER'],
        'angle_right_shank': ['RIGHT ANKLE', 'RIGHT KNEE'],
        'angle_left_shank': ['LEFT ANKLE', 'LEFT KNEE']
    }

    for angle_name, input_value in rectification_needed_items:
        
        # mapped_angle_name = map_angle_names(angle_name, batsman_type)
        correct_angle_name = correct_angle_name_converter(angle_name, batsman_type)
        
        if angle_name in reference_angles and angle_name in deviation_thresholds:
            reference_value = reference_angles[angle_name]
            accurate_threshold, minor_error_threshold = deviation_thresholds[angle_name]

            error = abs(input_value - reference_value)

            if error >2*minor_error_threshold:
                error_type = "large error"
                if input_value > reference_value:
                    error_description = "too wide"
                else:
                    error_description = "too narrow"
            elif minor_error_threshold < error <= 2 * minor_error_threshold:
                error_type = "minor error"
                if abs(input_value - reference_value) < accurate_threshold:
                    error_description = "narrow"
                else:
                    error_description = "wide"

            lower_bound = round(reference_value - minor_error_threshold)
            upper_bound = round(reference_value + minor_error_threshold)
            acceptable_range = f"{lower_bound} to {upper_bound}"
            
            # Provide feedback based on neighboring joints
            neighbors = neighboring_joints.get(angle_name, [])
            # feedback = check_neighboring_joints(angle_name, input_angles, neighbors)

            if error_description == "narrow" or error_description == "too narrow":
                correct_action = "Widen"
            elif error_description == "wide" or error_description == "too wide":
                correct_action = "Narrow"
            elif (error_description == "wide" or error_description == "too wide") and (correct_angle_name == 'Right Shank' or correct_angle_name == 'Right Shank'):
                correct_action = f"Move forword your {neighbors[1]}"
            elif error_description == "narrow" or error_description == "too narrow" and (correct_angle_name == 'Right Shank' or correct_angle_name == 'Right Shank'):
                correct_action = f"Move backward your {neighbors[1]}"
            else:
                correct_action = "adjust"
            message = {
                'angle name': correct_angle_name,
                'current angle value': round(input_value),
                'acceptable range': acceptable_range,
                'error type': error_type,
                'error description': error_description,
                'neighboring joints to change':f'{neighbors[0]} & {neighbors[1]}',
                'action': f'{correct_action} the angle'
            }

            rectifications.append(message)

    return rectifications

#check neighboring joints of the landmarks
def check_neighboring_joints(input_angles, neighbors):
    # Limit feedback to the first 2 neighboring joints
    feedback = []
    for neighbor in neighbors[:2]:  # Take only the first 2 neighbors
        if neighbor in input_angles:
            feedback.append(neighbor)
    
    return feedback

#angle name converter
def correct_angle_name_converter(angle_name, batsman_type):
    angle_name_mapping = {
        'angle_right_elbow': 'Right Elbow',
        'angle_left_elbow': 'Left Elbow',
        'angle_right_shoulder': 'Right Shoulder',
        'angle_left_knee': 'Left Knee',
        'angle_right_knee': 'Right Knee',
        'angle_right_hip_knee': 'Right Hip to Left Hip',
        'angle_left_hip_knee': 'Left Hip to Right Hip',
        'angle_right_hip_shoulder': 'Right Hip to Shoulder',
        'angle_right_shank': 'Right Shank',
        'angle_left_shank': 'Left Shank'
    }

    if batsman_type == 'left-hand':
        # Swap right with left for left-hand batsmen
        angle_name = angle_name.replace('right_', 'temp_').replace('left_', 'right_').replace('temp_', 'left_')

    return angle_name_mapping.get(angle_name, angle_name)

#Get average bat angle for the stroke
def get_average_bat_angle(shot_type):

    stats_directory = f'D:/Research/Cric-Boost/models/batsmen/stats/{shot_type} analysis.csv'
    reference_data = pd.read_csv(stats_directory)
    
    # Define the distance columns
    bat_angle = 'bat_angle'

    if bat_angle in reference_data.columns:
        # Calculate the average bat angle
        average_bat_angle = reference_data[bat_angle].mean()
        sd_bat_angle = reference_data[bat_angle].std()
        return average_bat_angle, sd_bat_angle
    else:
        average_bat_angle = None  
        return None
    
#Get average bat gap for the stroke based on the stroke
def get_average_bat_gap(shot_type):

    stats_directory = f'D:/Research/Cric-Boost/models/batsmen/stats/{shot_type} analysis.csv'
    reference_data = pd.read_csv(stats_directory)
    
    # Define the distance columns
    bat_gap_body = 'gap_leg_bat'

    if bat_gap_body in reference_data.columns:
        # Calculate the average bat angle
        average_bat_angle = reference_data[bat_gap_body].mean()
        return average_bat_angle
    else:
        average_bat_angle = None  
        return None