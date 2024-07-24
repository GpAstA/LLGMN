import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def save_results(output_dir, all_predictions, all_targets, all_weights, all_losses, avg_accuracy, conf_matrix):
    # 各予測結果と正解をCSVに保存
    df_predictions = pd.DataFrame({'Prediction': all_predictions, 'Target': all_targets})
    df_predictions.to_csv(os.path.join(output_dir, 'predictions_targets.csv'), index=False)

    # 各Foldごとの各変数の重みをCSVに保存
    df_weights = pd.DataFrame(np.vstack(all_weights), columns=[f'Weight_{i+1}' for i in range(all_weights[0].shape[1])])
    df_weights.to_csv(os.path.join(output_dir, 'fold_weights.csv'), index=False)

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
