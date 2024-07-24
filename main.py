import torch
from torch.utils.data import DataLoader
import optuna
from objective import objective
from data_processing.data_loader import load_data

def main():
    # fix seed and device
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # set each parameter
    n_component = 3
    n_epoch = 100
    batch_size = 32
    csv_file = 'LLGMN/input/230127_ATAL_all_嚥下障害有無_cFIM_mFIM.csv'  # CSVファイルのパスを指定

    # load data
    train_dataset, in_features, n_class = load_data(csv_file)

    # make dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_dataloader, in_features, n_class, n_component, n_epoch), n_trials=50)

    # 最適なハイパーパラメータの表示
    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print(f'  Accuracy: {trial.user_attrs["accuracy"]}%')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

if __name__ == '__main__':
    main()
