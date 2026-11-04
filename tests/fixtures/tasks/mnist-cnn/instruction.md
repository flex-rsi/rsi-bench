# MNIST classification under a training budget

Train a classifier for 28x28 handwritten digits from scratch and predict held-out images.

`model.py` must define

```python
def fit_predict(train_images, train_labels, images, device="cuda") -> array of int:
```

`train_images` is a uint8 array [50000, 28, 28], `train_labels` an int array [50000], `images`
the uint8 images to classify. The function trains from scratch and returns one predicted digit
per image. The evaluator calls it once per evaluation, on one GPU, inside the task container
(PyTorch with CUDA, numpy). Training plus prediction must finish within 15 minutes; nothing may be
downloaded (no network) and no pretrained weights may be used.

Metric: accuracy on the query images (higher is better). The training set and the dev query split
(images and labels) are in the dev data directory as `.npy` files. Use `run_experiment` with
`gpus: 1` to train on a GPU; `$RSIB_PYTHON` is the interpreter to use inside the container.
