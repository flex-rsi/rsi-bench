# Handwritten digit classification (8x8)

Classify 8x8 grey-scale images of handwritten digits (0-9).

`classifier.py` must define

```python
def fit_predict(train_x, train_y, x) -> list[int]:
```

`train_x` holds 900 training images as lists of 64 pixel values (0..16), `train_y` their digits,
and `x` the images to classify. Return one digit per image in `x`. The evaluator calls the
function in a separate process with the query images of the dev, validation or test split.

Available packages: the Python standard library, `numpy` and `scikit-learn`. CPU only. One call
must finish within five minutes.

Metric: accuracy on the query images (higher is better). The training set and the dev query split
(images and labels) are in the dev data directory.
