import torch


def accuracy(true_labels, logits):
    # logits are the raw model outputs
    preds = logits.argmax(-1)
    acc = (preds == true_labels.view_as(preds)).float().detach().cpu().numpy().mean()
    return acc