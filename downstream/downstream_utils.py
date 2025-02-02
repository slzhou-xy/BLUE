import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, f1_score, accuracy_score, precision_score


def mr(dists):
    targets = np.diag(dists)
    result = np.sum(np.greater_equal(dists.T, targets)) / dists.shape[0]
    return round(result, 5)


def hit_ratio(truth, pred, Ks):
    hit_K = {}
    for K in Ks:
        top_K_pred = pred[:, :K]
        hit = 0
        for i, pred_i in enumerate(top_K_pred):
            if truth[i] in pred_i:
                hit += 1
        hit_K[K] = round(hit / pred.shape[0], 5)
    return hit_K


def travel_time_evaluation(preds, labels):
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = preds * 60
    labels = labels * 60
    mae = mean_absolute_error(labels, preds)
    mape = mean_absolute_percentage_error(labels, preds)
    rmse = mean_squared_error(labels, preds) ** 0.5
    return {'MAE': round(mae, 5), 'RMSE': round(rmse, 5), 'MAPE': round(mape, 5)}


def multi_cls_evaluation(preds, truths, n_classes):
    preds = np.vstack(preds)
    truths = np.concatenate(truths)
    preds_label = np.argmax(preds, axis=-1)
    micro_f1 = f1_score(truths, preds_label, average='micro', labels=np.arange(n_classes).tolist())
    macro_f1 = f1_score(truths, preds_label, average='macro', labels=np.arange(n_classes).tolist())
    return {'Mi-F1': round(micro_f1, 5), 'Ma-F1': round(macro_f1, 5)}


def binary_cls_evaluation(preds, truths):
    preds = np.vstack(preds)
    truths = np.concatenate(truths)
    preds_label = np.argmax(preds, axis=-1)
    f1 = f1_score(truths, preds_label)
    accuracy = accuracy_score(truths, preds_label)
    precision = precision_score(truths, preds_label)

    return {'F1': round(f1, 5), 'Accuracy': round(accuracy, 5), 'Precision': round(precision, 5)}
