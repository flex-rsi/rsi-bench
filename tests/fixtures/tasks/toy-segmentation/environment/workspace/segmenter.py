"""Starter solution: one global threshold."""


def predict(train_images, train_masks, images):
    """Return one binary mask (list of lists of 0/1) per image."""
    return [[[1 if v > 0.5 else 0 for v in row] for row in img] for img in images]
