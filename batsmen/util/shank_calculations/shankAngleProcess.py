import numpy as np
from util.shank_calculations.shankAngleCalculate import shankAngleCalculate

def shankAngleCalculationProcess(ankle_foot, ankle_knee, reference):
    reference_point = reference
    shank_vector = ankle_foot - ankle_knee
    reference_vector = reference_point - ankle_foot
    angle = shankAngleCalculate(shank_vector, reference_vector)
    return angle
