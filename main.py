from data_processing.data_loader import load_data
from utils.utils import train_and_evaluate

# パラメータ設定
n_class = 2
n_component = 2
n_epoch = 100
K = 5
acc_thr = 0.5
target = r'C:\Users\ME-PC2\LLGMN-pytorch\LLGMN\input\230127_ATAL_all_嚥下障害有無_cFIM_mFIM.csv'

if __name__ == "__main__":
    x, y = load_data(target)
    input_dim = x.shape[1]
    
    train_and_evaluate(x, y, input_dim, n_class, n_component, n_epoch, K, acc_thr)