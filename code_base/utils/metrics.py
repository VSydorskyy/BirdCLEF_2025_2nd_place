import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "row_id") -> float:
    '''
    Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    return roc_auc_score(
        y_true=solution[scored_columns].values,
        y_score=submission[scored_columns].values,
        average="macro"
    )

def score_numpy(y_true: np.ndarray, y_pred: np.ndarray):
    scored_columns_mask = y_true.sum(axis=0) > 0

    y_true_filtered = y_true.T[scored_columns_mask].T
    y_pred_filtered = y_pred.T[scored_columns_mask].T

    return roc_auc_score(
        y_true=y_true_filtered,
        y_score=y_pred_filtered,
        average="macro"
    )
