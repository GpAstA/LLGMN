# utils/result_saving.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def save_results(output_dir, all_predictions, all_targets, all_model_weights, all_reducer_weights, all_losses, avg_accuracy, conf_matrix, feature_names, best_params, accuracies):
    # 各予測結果と正解をCSVに保存
    df_predictions = pd.DataFrame({'Prediction': all_predictions, 'Target': all_targets})
    df_predictions.to_csv(os.path.join(output_dir, 'predictions_targets.csv'), index=False)

    # 各Foldごとの各変数の重みをCSVに保存
    # model_weights_reshaped = []
    # for weights in all_model_weights:
    #     model_weights_reshaped.append(weights.flatten())
    # model_weights_reshaped = np.vstack(model_weights_reshaped)
    # df_model_weights = pd.DataFrame(model_weights_reshaped, columns=[f'Model_Weight_{name}' for name in feature_names])
    # df_model_weights.to_csv(os.path.join(output_dir, 'fold_model_weights.csv'), index=False)
    num_folds = len(all_model_weights)
    num_features = len(feature_names)
    model_weights_reshaped = np.array(all_model_weights).reshape((num_folds, num_features))
    df_model_weights = pd.DataFrame(model_weights_reshaped, columns=[f'Model_Weight_{name}' for name in feature_names])
    df_model_weights.to_csv(os.path.join(output_dir, 'fold_model_weights.csv'), index=False)
    reducer_weights_reshaped = np.array(all_reducer_weights).reshape((num_folds, num_features))
    df_reducer_weights = pd.DataFrame(reducer_weights_reshaped, columns=[f'Reducer_Weight_{name}' for name in feature_names])
    df_reducer_weights.to_csv(os.path.join(output_dir, 'fold_reducer_weights.csv'), index=False)


    # reducer_weights_reshaped = []
    # for weights in all_reducer_weights:
    #     reducer_weights_reshaped.append(weights.flatten())
    # reducer_weights_reshaped = np.vstack(reducer_weights_reshaped)
    # df_reducer_weights = pd.DataFrame(reducer_weights_reshaped, columns=[f'Reducer_Weight_{name}' for name in feature_names])
    # df_reducer_weights.to_csv(os.path.join(output_dir, 'fold_reducer_weights.csv'), index=False)

    # 混同行列をPNGに保存
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title(f'Confusion Matrix\nAccuracy: {avg_accuracy:.2f}')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 学習時の損失グラフをPNGに保存
    plt.figure()
    for fold_losses in all_losses:
        plt.plot(fold_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend([f'Fold {i+1}' for i in range(len(all_losses))])
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    # ハイパーパラメータと識別精度をCSVに保存
    df_hyperparams = pd.DataFrame(best_params)
    df_hyperparams['Accuracy'] = accuracies
    df_hyperparams.to_csv(os.path.join(output_dir, 'hyperparams_accuracy.csv'), index=False)
