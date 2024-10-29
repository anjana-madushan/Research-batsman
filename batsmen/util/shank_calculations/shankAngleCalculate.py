import numpy as np

def shankAngleCalculate(shank_vector, reference_vector):
    dot_product = np.dot(shank_vector, reference_vector)
    magnitude1 = np.linalg.norm(shank_vector)
    magnitude2 = np.linalg.norm(reference_vector)
    radians = np.arccos(dot_product / (magnitude1 * magnitude2))
    degrees = np.degrees(radians)
    return round(degrees, 3)