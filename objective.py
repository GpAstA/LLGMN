# src/objective.py

import torch
import torch.optim as optim
from model.through_net import ImprovedThroughNet
from model.LLGMN import LLGMN
from data_processing.data_loader import get_train_dataloader
import torch.nn as nn

def objective(trial):
    # Optunaでハイパーパラメータをサンプリング
    alpha = trial.suggest_float('alpha', 0.0, 1.0)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

    # モデルのインスタンス化
    input_shape = (100,)
    output_dim = 50
    n_class = 10
    n_component = 5

    reducer = ImprovedThroughNet(input_shape, output_dim, alpha, l1_ratio)
    LLGMN = LLGMN(output_dim, n_class, n_component)

    if torch.cuda.is_available():
        reducer = reducer.cuda()
        LLGMN = LLGMN.cuda()

    # オプティマイザの設定
    optimizer = optim.Adam(list(reducer.parameters()) + list(LLGMN.parameters()), lr=0.001)

    # データローダーの取得
    train_dataloader = get_train_dataloader(batch_size=32)

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()

    # トレーニングループ
    num_epochs = 10
    for epoch in range(num_epochs):
        reducer.train()
        LLGMN.train()
        total_loss = 0

        for inputs, targets in train_dataloader:
            optimizer.zero_grad()

            # フォワードパス
            reduced_output =    reducer(inputs)
            outputs = LLGMN(reduced_output)

            # 損失の計算
            loss = criterion(outputs, targets)

            # ElasticNetペナルティの追加
            loss += reducer.get_elasticnet_penalty()

            # バックプロパゲーションとパラメータ更新
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

    return loss.item()
