import torch


def calc_one_gram(predicted, target):
    count = 0.0
    total = 0.0
    tensor_p = predicted.reshape(-1)
    tensor_t = target.contiguous().view(-1)
    target_set = set(tensor_t.numpy())
    for item in tensor_p:
        total += 1.0
        if item in target_set and item > 3.0:
            count += 1.0

    return count / total
