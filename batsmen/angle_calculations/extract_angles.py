import numpy as np
import mediapipe as mp
import cv2

from util.shank_calculations.shankAngleCalculate import shankAngleCalculate
from util.shank_calculations.shankAngleProcess import shankAngleCalculationProcess

def calculate_angle(A, B, C):
    AB = A - B
    CB = C - B
    dot_product = np.dot(AB, CB)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_CB = np.linalg.norm(CB)
    cosine_angle = dot_product / (magnitude_AB * magnitude_CB)
    angle = np.degrees(np.arccos(cosine_angle))
    return round(angle, 3)

def extract_angles(image_np, batsman_type):
    mp_pose = mp.solutions.pose
    
    # Mapping of left and right landmarks based on batsman type
    landmark_mapping = {
        'right-hand': {
            'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER,
            'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'LEFT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW,
            'RIGHT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW,
            'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
            'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
            'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
            'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
            'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE,
            'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE,
            'RIGHT_WRIST': mp_pose.PoseLandmark.RIGHT_WRIST
        },
        'left-hand': {
            'LEFT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'RIGHT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER,
            'LEFT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW,
            'RIGHT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW,
            'LEFT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
            'RIGHT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
            'LEFT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
            'RIGHT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
            'LEFT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE,
            'RIGHT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE,
            'RIGHT_WRIST': mp_pose.PoseLandmark.LEFT_WRIST
        }
    }
    
    # Check if the batsman_type is valid
    if batsman_type not in landmark_mapping:
        raise ValueError(f"Unsupported batsman_type: {batsman_type}")
    
    # Get the correct mapping for the batsman type
    mapping = landmark_mapping[batsman_type]
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.2) as pose:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            # Extract keypoints based on batsman type
            points = {key: np.array([landmarks[landmark.value].x,
                                    landmarks[landmark.value].y,
                                    landmarks[landmark.value].z])
                      for key, landmark in mapping.items()}
                
            # Calculate angles
            angles = {
                'angle_right_elbow': calculate_angle(points['RIGHT_SHOULDER'], points['RIGHT_ELBOW'], points['RIGHT_WRIST']),
                'angle_left_elbow': calculate_angle(points['LEFT_SHOULDER'], points['LEFT_ELBOW'], points['RIGHT_WRIST']),
                'angle_right_shoulder': calculate_angle(points['RIGHT_HIP'], points['RIGHT_SHOULDER'], points['RIGHT_ELBOW']),
                'angle_left_knee': calculate_angle(points['LEFT_HIP'], points['LEFT_KNEE'], points['LEFT_ANKLE']),
                'angle_right_knee': calculate_angle(points['RIGHT_HIP'], points['RIGHT_KNEE'], points['RIGHT_ANKLE']),
                'angle_right_hip_knee': calculate_angle(points['RIGHT_KNEE'], points['RIGHT_HIP'], points['LEFT_HIP']),
                'angle_left_hip_knee': calculate_angle(points['LEFT_KNEE'], points['LEFT_HIP'], points['RIGHT_HIP']),
                'angle_right_hip_shoulder': calculate_angle(points['RIGHT_KNEE'], points['RIGHT_HIP'], points['RIGHT_SHOULDER']),
                'angle_right_shank': shankAngleCalculationProcess(points['RIGHT_ANKLE'], points['RIGHT_KNEE'], points['LEFT_ANKLE']),
                'angle_left_shank': shankAngleCalculationProcess(points['LEFT_ANKLE'], points['LEFT_KNEE'], points['RIGHT_ANKLE']),
            }
            
            return angles
        else:
            return None