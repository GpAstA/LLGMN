import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch
from model.LLGMN import LLGMN
from train import train_model
import optuna
from objective import objective



def main():
    # fix seed and device
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # set each parameter 
    in_features = 5
    n_class = 2
    n_component = 3
    n_epoch = 30
    batch_size = 32

    # generate datas
    x_train = torch.randn(100, 5)
    x_train[0:50, :] += 2
    y_train = torch.ones((100), dtype=int)
    y_train[0:50] -= 1

    # make dataset
    train_dataset = TensorDataset(x_train, y_train)

    # make dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

if __name__ == '__main__':
    main()