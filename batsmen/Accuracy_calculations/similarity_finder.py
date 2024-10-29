import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def load_reference_data(shot_type):

    stroke = ''
    if shot_type == 'forward_defence':
        stroke = 'forward defence'
    elif shot_type == 'forward_drive':
        stroke = 'forward drive'
    if shot_type == 'backfoot_defence':
        stroke = 'backfoot defence'
    if shot_type == 'backfoot_drive':
        stroke = 'backfoot drive'
    stats_directory = f'D:/Research/Cric-Boost/models/batsmen/stats/{stroke} analysis.csv'
    reference_data = pd.read_csv(stats_directory)
    
    # Define the distance columns
    distance_columns = [
        'left_shoulder_right_shoulder',
        'right_shoulder_right_elbow',
        'right_shoulder_right_hip',
        'left_hip_right_hip',
        'right_hip_right_knee',
        'right_knee_right_ankle'
    ]
    return reference_data, distance_columns

def find_closest_match(user_distances, shot_type):
    # print(user_distances, shot_type)
    shot_type = shot_type.replace(' ', '_')

    # Load the reference data for the given shot type
    reference_data, distance_columns = load_reference_data(shot_type)
    
    X_train = reference_data[distance_columns].values
    X_query = np.array([list(user_distances.values())])
    
    k = int(np.sqrt(X_train.shape[0]))  # Square Root of N rule
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn.fit(X_train)
    distances, indices = knn.kneighbors(X_query)
    closest_matches = reference_data.iloc[indices[0]]
    
    return closest_matches.drop(columns=distance_columns + ['label', 'image_name'])
