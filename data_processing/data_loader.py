import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return x, y
