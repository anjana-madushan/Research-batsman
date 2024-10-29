import numpy as np
import mediapipe as mp
import cv2

def head_position_check_fd(image_np,  batsman_type):
    mp_pose = mp.solutions.pose

    # Mapping of left and right landmarks based on batsman type
    landmark_mapping = {
        'left-hand': {
            'LEFT_EAR': mp_pose.PoseLandmark.LEFT_EAR,
            'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
            'NOSE': mp_pose.PoseLandmark.NOSE,
        },
        'right-hand': {
            'RIGHT_EAR': mp_pose.PoseLandmark.RIGHT_EAR,
            'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
            'NOSE': mp_pose.PoseLandmark.NOSE,
        }
    }

    if batsman_type not in landmark_mapping:
        raise ValueError("Invalid batsman type. Choose either 'left-hand' or 'right-hand'.")

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
            
            # Calculate the midpoint between the ear and the nose
            if batsman_type == 'left-hand':
                left_ear = points['LEFT_EAR']
                nose = points['NOSE']
                midpoint = (left_ear + nose) / 2
                if midpoint[0]<=points['RIGHT_KNEE'][0]+(nose[0]-left_ear[0]):
                    return True
                else:
                    return False
            elif batsman_type == 'right-hand':
                right_ear = points['RIGHT_EAR']
                nose = points['NOSE']
                midpoint = (right_ear + nose) / 2
                if midpoint[0]<=points['LEFT_KNEE'][0]+(nose[0]-right_ear[0]):
                    return True
                else:
                    return False
        
    return None
