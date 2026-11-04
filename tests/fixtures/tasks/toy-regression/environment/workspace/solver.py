"""Starter solution: a straight line fitted by least squares."""


def fit_predict(train_x: list[float], train_y: list[float], query_x: list[float]) -> list[float]:
    n = len(train_x)
    mean_x = sum(train_x) / n
    mean_y = sum(train_y) / n
    sxx = sum((x - mean_x) ** 2 for x in train_x)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(train_x, train_y))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    return [intercept + slope * x for x in query_x]
