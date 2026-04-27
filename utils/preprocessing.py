import numpy as np

def preprocess_input(age, gender, heart_rate, steps):
    gender_val = 1 if gender == "Male" else 0
    return np.array([[age, gender_val, heart_rate, steps]])
