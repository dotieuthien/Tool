import torch
from torch.nn import functional


def compute_permute_loss(output_a, output_b, positive_pairs, t=0.01):
    affinity = torch.bmm(output_a, torch.transpose(output_b, 2, 1)) / t
    total_match = 0
    loss = None

    for i in range(0, output_a.shape[1]):
        match_count = positive_pairs[:, i, :].sum()
        if match_count == 0:
            continue

        total_match += match_count
        target = positive_pairs[0, i, :]
        target = torch.nonzero(target == 1)[0]

        if loss is None:
            loss = functional.cross_entropy(affinity[:, i, :], target)
        else:
            loss += functional.cross_entropy(affinity[:, i, :], target)

    loss = loss / total_match
    return loss
