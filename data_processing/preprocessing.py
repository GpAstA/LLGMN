from sklearn.preprocessing import StandardScaler
import numpy as np

def standardize(train_data, test_data):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data, scaler

def convert_labels(labels, n_class):
    return np.eye(n_class)[labels]
