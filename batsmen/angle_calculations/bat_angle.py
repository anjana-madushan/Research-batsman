import numpy as np
import mediapipe as mp
import cv2

def calculate_angle(A, B, C):
    AB = A - B
    CB = C - B
    dot_product = np.dot(AB, CB)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_CB = np.linalg.norm(CB)
    cosine_angle = dot_product / (magnitude_AB * magnitude_CB)
    angle = np.degrees(np.arccos(cosine_angle))
    return round(angle, 3)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def get_knee_positions(image_np, batting_type):
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.2) as pose:
      # Convert image to RGB
      image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
      results = pose.process(image)

      # Check if landmarks are detected
      if results.pose_world_landmarks:
          # Get the landmarks as a list
          landmarks = results.pose_world_landmarks.landmark
          
          # Extract the coordinates for the right and left knees 
          right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
          left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

          # Convert the landmark coordinates to a tuple (x, y)
          if batting_type == 'left-hand':
             left_knee_position = (int(left_knee.x * image_np.shape[1]), int(left_knee.y * image_np.shape[0]))
             return left_knee_position
          elif batting_type == 'right-hand':
            right_knee_position = (int(right_knee.x * image_np.shape[1]), int(right_knee.y * image_np.shape[0]))
            return right_knee_position
      else:
          return None, None


def calculate_bat_angle(image_np, bat_bbox, stroke, batting_type):
    try:
      position = get_knee_positions(image_np, batting_type)

      # Calculate the bat's midpoint
      bat_mid_x = (bat_bbox[0] + bat_bbox[2]) // 2
      bat_mid_y = (bat_bbox[1] + bat_bbox[3]) // 2
      bat_mid_point = np.array([bat_mid_x, bat_mid_y])

      # Create a point for horizontal reference (ground level)
      knee_position = position
      reference_point = np.array([knee_position[0], bat_mid_y])

      # Top corners of the bounding box for the bat
      top_left = np.array([bat_bbox[0], bat_bbox[1]])
      top_right = np.array([bat_bbox[2], bat_bbox[1]])

      selected_angle = None
              
      shot_type = stroke.strip().lower()
      if shot_type in ["backfoot defence", "forward defence"]:
        # Calculate rpm (angle from r to p, and p to m)
        selected_angle = calculate_angle(reference_point, bat_mid_point, top_right)

      elif shot_type in ["backfoot drive", "forward drive"]:
        # Calculate rpn (angle from r to p, and p to n)
        selected_angle = calculate_angle(reference_point, bat_mid_point, top_left)

      return selected_angle
    except Exception as e:
        output_data = {
            'error': str(e)
        }
        return output_data