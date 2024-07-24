import torch
import torch.optim as optim
import torch.nn as nn
from model.LLGMN import LLGMN
from model.elasticnet_reducer import ElasticNetDimensionReducer
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

def objective(trial, dataset, in_features, n_class, n_component, n_epoch, n_splits):
    # ハイパーパラメータのサンプリング
    alpha = trial.suggest_float('alpha', 0.0, 1.0)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)  # 学習率を制約

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    labels = [dataset[i][1] for i in range(len(dataset))]

    total_test_loss = 0.0
    total_accuracy = 0.0

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        # print(f"Fold {fold+1}/{n_splits}")

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_subset, batch_size=32, shuffle=False)

        # モデルのインスタンス化
        reducer = ElasticNetDimensionReducer(in_features, alpha, l1_ratio).to(device)
        model = LLGMN(in_features, n_class, n_component).to(device)  # 次元削減後の次元数をreduced_dimに設定
        optimizer = optim.Adam(list(reducer.parameters()) + list(model.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

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

            avg_train_loss = total_loss / len(train_dataloader)
            # print(f'Epoch {epoch+1}/{n_epoch}, Train Loss: {avg_train_loss}')
        
         # テストループ
        model.eval()
        reducer.eval()
        fold_test_loss = 0.0
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
                fold_test_loss += loss.item()

                # 精度の計算
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_fold_test_loss = fold_test_loss / len(test_dataloader)
        accuracy = 100 * correct / total
        print(f'Fold {fold+1}/{n_splits}, Test Loss: {avg_fold_test_loss}, Accuracy: {accuracy}%')

        total_test_loss += avg_fold_test_loss
        total_accuracy += accuracy

    avg_test_loss = total_test_loss / n_splits
    avg_accuracy = total_accuracy / n_splits

    print(f'Average Test Loss: {avg_test_loss}, Average Accuracy: {avg_accuracy}%')

    # 結果としてテスト損失を返すが、精度も記録する
    trial.set_user_attr('accuracy', avg_accuracy)
    return avg_test_loss
