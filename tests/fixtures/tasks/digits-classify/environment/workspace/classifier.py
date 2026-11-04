"""Starter solution: nearest class centroid."""

import numpy as np


def fit_predict(train_x, train_y, x):
    """train_x: [N, 64] pixel values 0..16, train_y: [N] digits, x: [M, 64] -> [M] predicted digits."""
    train_x, train_y, x = np.asarray(train_x, float), np.asarray(train_y), np.asarray(x, float)
    classes = np.unique(train_y)
    centroids = np.stack([train_x[train_y == c].mean(axis=0) for c in classes])
    dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    return classes[dist.argmin(axis=1)].tolist()
