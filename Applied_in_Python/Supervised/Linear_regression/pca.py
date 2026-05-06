import pandas as pd
import numpy as np


def reduce_dim_pca(data: pd.DataFrame, line: np.array):
    mean = data.mean()
    centered_d = data - mean
    centered_l = line - mean

    # computing the cov of the data
    cov = 1 /(len(data) - 1) * np.dot(np.transpose(centered_d.to_numpy()), centered_d.to_numpy())
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    w = eigvecs[:, idx[:2]]

    projected_d = centered_d @ w
    projected_l = centered_l @ w

    return projected_d, projected_l
