import torch
import torch.nn.functional
from loader.features_utils import convert_to_stacked_mask
from models.utils import gather_by_label_matrix, gather_by_one_hot_labels


def check_label_pooling():
    out = torch.zeros([1, 10, 100, 100], dtype=torch.float)
    labels = torch.zeros([1, 100, 100], dtype=torch.int64)

    out[0, :5, 10, 10] = 5.0
    out[0, 5:, 10, 10] = 8.0
    out[0, :, 70, 70] = 5.0

    labels[0, 10:30, 10:30] = 1
    labels[0, 50:75, 50:75] = 3

    output = gather_by_label_matrix(out, labels)
    print(output[0, 0])
    print(output[0, 1])
    print(output[0, 2])
    print(torch.nn.functional.cosine_similarity(output[:, 0, :], output[:, 1, :], dim=-1))

    print()

    labels = torch.tensor(convert_to_stacked_mask(labels[0].cpu().numpy())).unsqueeze(dim=0).float()
    output = gather_by_one_hot_labels(out, labels)
    print(output[0, 0])
    print(output[0, 1])
    print(output[0, 2])
    print(torch.nn.functional.cosine_similarity(output[:, 0, :], output[:, 1, :], dim=-1))


if __name__ == "__main__":
    check_label_pooling()
