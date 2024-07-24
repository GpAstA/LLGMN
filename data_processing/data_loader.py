import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

def load_data(csv_file):
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)

    # 入力データとターゲットデータに分割
    x_data = df.iloc[:, :-1].values
    y_data = df.iloc[:, -1].values

    # scaler = StandardScaler()
    # x_data = scaler.fit_transform(x_data)
    x_data = Standardization(x_data)

    # PyTorchのテンソルに変換
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)

    # 特徴量の数とクラス数を計算
    in_features = x_tensor.shape[1]
    n_class = len(torch.unique(y_tensor))

    # データセットを作成
    dataset = TensorDataset(x_tensor, y_tensor)
    return dataset, in_features, n_class


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

def Standardization(train):##
    ave = np.mean(train,axis=0)
    std = np.std(train,axis=0,ddof=1)
    for i in range(len(std)):
        if std[i] == 0:
            std[i] = 1000000
    scaler = (train - ave)/std
    return scaler