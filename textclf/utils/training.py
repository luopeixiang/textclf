import torch


def cal_accuracy(logits, labels):
    predicts = torch.argmax(logits, dim=1)
    acc = torch.mean((predicts == labels).float())
    return acc
