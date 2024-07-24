import torch
import torch.optim as optim
import torch.nn as nn
from model.LLGMN import LLGMN
from model.elasticnet_reducer import ElasticNetDimensionReducer

def objective(trial, train_dataloader, test_dataloader, in_features, n_class, n_component, n_epoch):
    # ハイパーパラメータのサンプリング
    alpha = trial.suggest_float('alpha', 0.0, 1.0)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)  # 学習率を制約

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルのインスタンス化
    reducer = ElasticNetDimensionReducer(in_features, 50, alpha, l1_ratio).to(device)
    model = LLGMN(50, n_class, n_component).to(device)  # 次元削減後の次元数を50に設定
    optimizer = optim.Adam(list(reducer.parameters()) + list(model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0

    # トレーニングループ
    for epoch in range(n_epoch):
        model.train()
        reducer.train()
        total_loss = 0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # フォワードパス
            reduced_inputs = reducer(inputs)
            outputs = model(reduced_inputs)

            # 損失の計算
            loss = criterion(outputs, targets)
            # ElasticNet正則化の追加
            loss += reducer.elasticnet_penalty()

            # バックプロパゲーションとパラメータ更新
            loss.backward()

            # 勾配のクリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # # 精度の計算
            # _, predicted = torch.max(outputs.data, 1)
            # total += targets.size(0)
            # correct += (predicted == targets).sum().item()

        avg_loss = total_loss / len(train_dataloader)
        # accuracy = 100 * correct / total
        # print(f'Epoch {epoch+1}/{n_epoch}, Loss: {avg_loss}, Accuracy: {accuracy}%')

        # # ベスト精度の更新
        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
    
    # テストループ
    model.eval()
    reducer.eval()
    total_test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # フォワードパス
            reduced_inputs = reducer(inputs)
            outputs = model(reduced_inputs)

            # 損失の計算
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()

            # 精度の計算
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    accuracy = 100 * correct / total
    # print(f'Test Loss: {avg_test_loss}, Accuracy: {accuracy}%')


    # 結果として損失を返すが、精度も記録する
    trial.set_user_attr('accuracy', accuracy)
    return avg_test_loss
