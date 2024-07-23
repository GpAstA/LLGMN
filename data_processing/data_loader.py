import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return x, y

## デモ用データ作成
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_train_dataloader(batch_size=32):
    # ダミーデータを作成（例）
    inputs = torch.randn(1000, 100)  # サンプル数1000、特徴量100
    targets = torch.randint(0, 10, (1000,))  # クラス数10のターゲット

    # TensorDatasetとDataLoaderを使用してデータセットを作成
    dataset = TensorDataset(inputs, targets)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader