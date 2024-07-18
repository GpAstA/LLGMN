import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from data_processing.preprocessing import standardize, convert_labels
from model.model import build_model

def train_and_evaluate(x, y, input_dim, n_class, n_component, n_epoch, K, acc_thr):
    skf = StratifiedKFold(n_splits=K, random_state=0, shuffle=True)
    
    for train_index, test_index in skf.split(x, y):
        x_train_full, x_test = x[train_index], x[test_index]
        y_train_full, y_test = y[train_index], y[test_index]

        x_train_full, x_test, scaler = standardize(x_train_full, x_test)

        # トレーニングデータをさらに分割して検証データを作成
        skf_val = StratifiedKFold(n_splits=K, random_state=0, shuffle=True)
        train_index, val_index = next(skf_val.split(x_train_full, y_train_full))

        x_train, x_val = x_train_full[train_index], x_train_full[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]

        # convert_labelsを分割後に適用
        y_train = convert_labels(y_train, n_class)
        y_val = convert_labels(y_val, n_class)
        y_test = convert_labels(y_test, n_class)

        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = build_model(input_dim, n_class, n_component)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(n_epoch):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.view(-1, 1)
                    targets = targets.argmax(dim=1).view(-1, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                
                val_accuracy = correct / total
                print(f'Epoch {epoch+1}/{n_epoch}, Validation Accuracy: {val_accuracy}')
                if val_accuracy > acc_thr:
                    break

        # 最後にテストデータで評価
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.view(-1, 1)
                targets = targets.argmax(dim=1).view(-1, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            test_accuracy = correct / total
            print(f'Test Accuracy: {test_accuracy}')
