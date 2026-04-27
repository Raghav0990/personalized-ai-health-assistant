import numpy as np
from sklearn.ensemble import RandomForestClassifier

def load_model():
    X = np.random.rand(100,4)
    y = (X[:,0] + X[:,2] > 1).astype(int)
    model = RandomForestClassifier()
    model.fit(X,y)
    return model

def predict(model, features):
    prob = model.predict_proba(features)[0][1]
    return ("HIGH RISK" if prob>0.5 else "LOW RISK", prob)
