"""Starter solution: a linear classifier trained for one epoch with plain SGD."""

import numpy as np
import torch
import torch.nn as nn


def fit_predict(train_images, train_labels, images, device="cuda"):
    """train_images: uint8 [N, 28, 28], train_labels: int [N], images: uint8 [M, 28, 28].

    Trains from scratch and returns the predicted digit for each image in `images`.
    """
    torch.manual_seed(0)
    x = torch.as_tensor(train_images, dtype=torch.float32, device=device).flatten(1) / 255.0
    y = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    model = nn.Linear(28 * 28, 10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(1):
        perm = torch.randperm(len(x), device=device)
        for i in range(0, len(x), 256):
            idx = perm[i:i + 256]
            loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        q = torch.as_tensor(images, dtype=torch.float32, device=device).flatten(1) / 255.0
        return model(q).argmax(dim=1).cpu().numpy()
