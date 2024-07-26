import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from model.LLGMN import LLGMN
from model.elasticnet_reducer import ElasticNetDimensionReducer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import Subset
from utils.result_saving import save_results
from datetime import datetime

def train_model(train_subset, test_subset, in_features, n_class, n_component, alpha, l1_ratio, lr, n_epoch, device):
    train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_subset, batch_size=32, shuffle=False)

    train_labels = np.array([train_subset[i][1] for i in range(len(train_subset))])
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    reducer = ElasticNetDimensionReducer(in_features, alpha, l1_ratio).to(device)
    model = LLGMN(in_features, n_class, n_component).to(device)
    optimizer = optim.Adam(list(reducer.parameters()) + list(model.parameters()), lr=0.001)
    # optimizer = optim.SGD(list(reducer.parameters()) + list(model.parameters()), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True) 
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()  # loss function. Don't use CrossEntropyLoss with LLGMN.

    fold_losses = []
    best_loss = float('inf')
    epochs_no_improve = 0
    patience=5

    for epoch in range(n_epoch):
        model.train()
        reducer.train()
        total_loss = 0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            reduced_inputs = reducer(inputs)
            outputs = model(reduced_inputs)

            loss = criterion(outputs, targets)
            loss += reducer.elasticnet_penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        fold_losses.append(avg_train_loss)

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    reducer.reduce_dimension()

    model.eval()
    reducer.eval()
    fold_predictions = []
    fold_targets = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            reduced_inputs = reducer(inputs)
            outputs = model(reduced_inputs)

            _, predicted = torch.max(outputs.data, 1)
            fold_predictions.extend(predicted.cpu().numpy())
            fold_targets.extend(targets.cpu().numpy())
    
    fold_accuracy = accuracy_score(fold_targets, fold_predictions)
    model_weights = model.weight.cpu().detach().numpy().flatten()
    reducer_weights = reducer.dense.weight.cpu().detach().numpy().flatten()

    return fold_losses, fold_predictions, fold_targets, model_weights, reducer_weights, fold_accuracy


def training_and_evaluation(csv_file, dataset, in_features, n_class, n_component, best_alpha, best_l1_ratio, best_lr, n_epoch, n_splits, best_params, accuracies):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    labels = [dataset[i][1] for i in range(len(dataset))]

    all_predictions = []
    all_targets = []
    all_model_weights = []
    all_reducer_weights = []
    all_losses = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    csv_filename = os.path.splitext(os.path.basename(csv_file))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'output/{csv_filename}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Get feature names from CSV
    df = pd.read_csv(csv_file)
    feature_names = df.columns[:-1].tolist()  # Assuming the last column is the target variable

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        print(f"Fold {fold+1}/{n_splits}")

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        fold_losses, fold_predictions, fold_targets, model_weights, reducer_weights, fold_accuracy = train_model(
            train_subset, test_subset, in_features, n_class, n_component, best_alpha, best_l1_ratio, best_lr, n_epoch, device
        )

        all_losses.append(fold_losses)
        all_predictions.extend(fold_predictions)
        all_targets.extend(fold_targets)
        all_model_weights.append(model_weights)
        all_reducer_weights.append(reducer_weights)

    avg_accuracy = accuracy_score(all_targets, all_predictions)
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    print(f'Final Accuracy: {avg_accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    feature_names = pd.read_csv(csv_file).columns[:-1]
    save_results(output_dir, all_predictions, all_targets, all_model_weights, all_reducer_weights, all_losses, avg_accuracy, conf_matrix, feature_names, best_params, accuracies)
