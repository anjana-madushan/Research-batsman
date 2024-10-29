import numpy as np
import mediapipe as mp
import cv2

def head_knee_toe_position_check_fdr(image,  batsman_type):
    mp_pose = mp.solutions.pose

    # Mapping of left and right landmarks based on batsman type
    landmark_mapping = {
        'left-hand': {
            'LEFT_EAR': mp_pose.PoseLandmark.LEFT_EAR,
            'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
            'NOSE': mp_pose.PoseLandmark.NOSE,
            'RIGHT_FOOT_INDEX':mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        },
        'right-hand': {
            'RIGHT_EAR': mp_pose.PoseLandmark.RIGHT_EAR,
            'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
            'NOSE': mp_pose.PoseLandmark.NOSE,
            'LEFT_FOOT_INDEX':mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        }
    }

    if batsman_type not in landmark_mapping:
        raise ValueError("Invalid batsman type. Choose either 'left-hand' or 'right-hand'.")

    mapping = landmark_mapping[batsman_type]
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.2) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            # Extract keypoints based on batsman type
            points = {key: np.array([landmarks[landmark.value].x,
                                     landmarks[landmark.value].y,
                                     landmarks[landmark.value].z])
                      for key, landmark in mapping.items()}
            
            threshold = 0
            # Calculate the midpoint between the ear and the nose
            if batsman_type == 'left-hand':
                ear = points['LEFT_EAR']
                knee = points['RIGHT_KNEE']
                foot_index = points['RIGHT_FOOT_INDEX']
                nose = points['NOSE']
                threshold = (nose[0]-ear[0])
            else:  # right-hand batsman
                ear = points['RIGHT_EAR']
                knee = points['LEFT_KNEE']
                foot_index = points['LEFT_FOOT_INDEX']
                nose = points['NOSE']
                threshold = (nose[0]-ear[0])

            midpoint = (ear + nose) / 2

            # Check alignment of each pair and store results in a dictionary
            alignment_status = {
                  'midpoint_knee_aligned': abs(midpoint[0] - knee[0]) <= threshold,
                  'knee_foot_index_aligned': abs(knee[0] - foot_index[0]) <= threshold
              }

            return alignment_status
    return None
