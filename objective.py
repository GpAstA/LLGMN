import torch
import torch.optim as optim
import torch.nn as nn
from model.LLGMN import LLGMN
from torch.utils.data import TensorDataset, DataLoader


def objective(trial):
    # ハイパーパラメータのサンプリング
    alpha = trial.suggest_float('alpha', 0.0, 1.0)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

    # ハイパーパラメータの宣言
    in_features = 5
    n_class = 2
    n_component = 3
    n_epoch = 30
    batch_size = 32

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # データの生成
    x_train = torch.randn(100, 5)
    x_train[0:50, :] += 2
    y_train = torch.ones((100), dtype=int)
    y_train[0:50] -= 1

    # データセットとデータローダーの作成
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # モデルのインスタンス化
    model = LLGMN(in_features, n_class, n_component).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # トレーニングループ
    for epoch in range(n_epoch):
        model.train()
        total_loss = 0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # フォワードパス
            outputs = model(inputs)

            # 損失の計算
            loss = criterion(outputs, targets)

            # バックプロパゲーションとパラメータ更新
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{n_epoch}, Loss: {avg_loss}')

    return avg_loss
