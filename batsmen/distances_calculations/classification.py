import numpy as np
import mediapipe as mp
import cv2

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def extract_distances(image_np, batsman_type):
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
            'NOSE':mp_pose.PoseLandmark.NOSE
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
            'NOSE':mp_pose.PoseLandmark.NOSE
        }
    }
    
    # Get the correct mapping for the batsman type
    mapping = landmark_mapping[batsman_type]
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.35) as pose:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            # Extract keypoints
            points = {}
            for key, landmark in mapping.items():
                points[key] = np.array([
                    landmarks[landmark.value].x,
                    landmarks[landmark.value].y,
                    landmarks[landmark.value].z
                ])
            # Compute distances between keypoints
            data = {               
                'left_shoulder_right_shoulder': calculate_distance(points['LEFT_SHOULDER'], points['RIGHT_SHOULDER']),
                'left_shoulder_left_elbow':calculate_distance(points['LEFT_SHOULDER'], points['LEFT_ELBOW']),
                'right_shoulder_right_elbow':calculate_distance(points['RIGHT_SHOULDER'], points['RIGHT_ELBOW']),
                'left_hip_right_hip':calculate_distance(points['LEFT_HIP'], points['RIGHT_HIP']),
                'left_knee_right_knee': calculate_distance(points['LEFT_KNEE'], points['RIGHT_KNEE']),
                'left_ankle_right_ankle': calculate_distance(points['LEFT_ANKLE'], points['RIGHT_ANKLE']),
                'left_knee_right_elbow': calculate_distance(points['LEFT_KNEE'], points['RIGHT_ELBOW']),
                'right_shoulder_left_knee':calculate_distance(points['RIGHT_SHOULDER'], points['LEFT_KNEE']),
                'right_hip_left_ankle':calculate_distance(points['RIGHT_HIP'], points['LEFT_ANKLE']),
                'right_knee_left_ankle': calculate_distance(points['RIGHT_KNEE'], points['LEFT_ANKLE']),
                'left_knee_right_ankle': calculate_distance(points['LEFT_KNEE'], points['RIGHT_ANKLE']),
                'nose_left_knee':calculate_distance(points['NOSE'], points['LEFT_KNEE']),
                'nose_left_ankle':calculate_distance(points['NOSE'], points['LEFT_ANKLE']),
            }
            return data
        else:
            return None