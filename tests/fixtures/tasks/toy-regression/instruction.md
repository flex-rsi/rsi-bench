# Toy regression

Predict a smooth one-dimensional function from noisy samples.

`solver.py` must define

```python
def fit_predict(train_x: list[float], train_y: list[float], query_x: list[float]) -> list[float]:
```

It receives the training set (200 points, x in [-2, 2]) and returns one finite prediction per
query point. The evaluator calls it in a separate process with the query points of the dev,
validation or test split. Standard library only; one call must finish within five minutes.

Metric: mean squared error on the query points (lower is better). The training data and the dev
query split (inputs and labels) are in the dev data directory.
