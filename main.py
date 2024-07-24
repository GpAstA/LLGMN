import torch
from torch.utils.data import DataLoader, random_split
import optuna
from objective import objective
from data_processing.data_loader import load_data
from model_training import training_and_evaluation

def main():
    # fix seed and device
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # set each parameter
    n_component = 3
    n_epoch = 40
    n_splits = 5
    n_trials = 5
    csv_file = 'LLGMN/input/230127_ATAL_all_嚥下障害有無_cFIM_mFIM.csv'  # CSVファイルのパスを指定

    # load data
    dataset, in_features, n_class = load_data(csv_file)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, dataset, in_features, n_class, n_component, n_epoch, n_splits), n_trials=n_trials)


    # 最適なハイパーパラメータの表示
    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print(f'  Accuracy: {trial.user_attrs["accuracy"]}%')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    
    best_alpha = trial.params['alpha']
    best_l1_ratio = trial.params['l1_ratio']
    best_lr = trial.params['lr']

    training_and_evaluation(csv_file, dataset, in_features, n_class, n_component, best_alpha, best_l1_ratio, best_lr, n_epoch, n_splits)

if __name__ == '__main__':
    main()
