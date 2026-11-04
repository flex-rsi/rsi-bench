# Toy segmentation

Segment bright disks in small noisy grey-scale images with uneven illumination.

`segmenter.py` must define

```python
def predict(train_images, train_masks, images) -> list[list[list[int]]]:
```

Images are 24x24 lists of floats. Masks use 1 for disk pixels and 0 for background. The function
receives 30 labelled training images and must return one mask per query image, with the same
shape. Standard library only; one call must finish within five minutes.

Metric: mean IoU over the two classes (disk, background) on the query images, higher is better.
The training set and the dev query split (images and masks) are in the dev data directory.
