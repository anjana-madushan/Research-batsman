import torch
import mediapipe as mp
import numpy as np
import cv2

def get_landmark_coordinate(imagenp, batsman_type, stroke):
    
    try:
        mp_pose = mp.solutions.pose
        # Mapping of left and right landmarks based on batsman type
        landmark_mapping = {
            'left-hand': {
                'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE,
                'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
                'NOSE': mp_pose.PoseLandmark.NOSE,
                'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
                'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
            },
            'right-hand': {
                'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE,
                'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
                'NOSE': mp_pose.PoseLandmark.NOSE,
                'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
                'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
            }
        }
        if batsman_type not in landmark_mapping:
            raise ValueError("Invalid batsman type. Choose either 'left-hand' or 'right-hand'.")

        mapping = landmark_mapping[batsman_type]
        with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.35) as pose:
 
            image_rgb = cv2.cvtColor(imagenp, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_world_landmarks is not None:
                landmarks = results.pose_world_landmarks.landmark
                
                # Extract keypoints based on batsman type
                points = {key: np.array([landmarks[landmark.value].x,
                                         landmarks[landmark.value].y,
                                         landmarks[landmark.value].z])
                          for key, landmark in mapping.items()}
                
                # Calculate the reference point
                if stroke in ['backfoot defence', 'backfoot drive']:
                    left_hip = points['LEFT_HIP']
                    right_hip = points['RIGHT_HIP']
                    midpoint = (left_hip + right_hip) / 2
                    return midpoint
                elif stroke in ['forward defence', 'forward drive']:
                    if batsman_type == 'left-hand':
                        right_knee = points['RIGHT_KNEE']
                        right_ankle = points['RIGHT_ANKLE']
                        midpoint = (right_knee + right_ankle) / 2
                        return midpoint
                    elif batsman_type == 'right-hand':
                        left_knee = points['LEFT_KNEE']
                        left_ankle = points['LEFT_ANKLE']
                        midpoint = (left_knee + left_ankle) / 2
                        return midpoint
            else:
                raise ValueError("Pose landmarks not detected.")
    
    except Exception as e:
        print(f"Error in get_landmark_coordinate: {e}")
        return None

def calculate_bat_gap(image_np, bat_bbox, stroke, batting_type):
    try:
        # Load MiDaS model for depth estimation
        positions = get_landmark_coordinate(image_np, batting_type, stroke)

        if positions is None:
            raise ValueError("Invalid positions returned from get_landmark_coordinate.")


        midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
        midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        midas_model.eval()

        # Convert to RGB if the image has an alpha channel (4 channels)
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        elif image_np.shape[2] == 1:  # Handle grayscale images
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        input_midas = midas_transform(image_np).unsqueeze(0)

        if input_midas.shape[1] == 1:
            input_midas = input_midas.squeeze(1)  # Correct indentation here

            with torch.no_grad():
                depth_map = midas_model(input_midas)
                depth_map = depth_map.squeeze().cpu().numpy()

                original_height, original_width = image_np.shape[:2]
                x_scale, y_scale = depth_map.shape[1] / original_width, depth_map.shape[0] / original_height

                # Calculate bat midpoint
                bat_midpoint = np.array([(bat_bbox[0] + bat_bbox[2]) / 2, (bat_bbox[1] + bat_bbox[3]) / 2])
                bat_x_center, bat_y_center = int(bat_midpoint[0] * x_scale), int(bat_midpoint[1] * y_scale)
                bat_depth = depth_map[bat_y_center, bat_x_center]

                # Scale the reference point midpoint coordinates for depth map indexing
                reference_x = int(positions[0] * original_width)
                reference_y = int(positions[1] * original_height)

                # Ensure scaled coordinates are within the bounds of the depth map
                reference_x = min(max(reference_x, 0), depth_map.shape[1] - 1)
                reference_y = min(max(reference_y, 0), depth_map.shape[0] - 1)

                reference_midpoint_depth = depth_map[reference_y, reference_x]
                gap_bat = abs(reference_midpoint_depth - bat_depth)

                return gap_bat

    except Exception as e:
        print(f"Error in calculate_bat_gap: {e}")
        return None
