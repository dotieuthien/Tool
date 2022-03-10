import torch
from models.utils import cosine_distance


def compute_infoNCE_loss(output_a, output_b, colors_a, colors_b, use_color=True):
    pos_distances, neg_distances = [], []
    colors_b = colors_b[0, ...]

    assert (len(output_a.shape) == 3) and (len(output_b.shape) == 3), f"expected output with 3 channels"

    MI = []
    for index_a in range(output_a.shape[1]):
        # fetch consider component features
        item_a = output_a[:, index_a, :]
        item_a = item_a.unsqueeze(1).repeat([1, output_b.shape[1], 1])
        current_color = colors_a[0, index_a, :]

        # measure distance to all components
        distances = cosine_distance(item_a, output_b)
        assert not torch.isnan(distances).any()

        # filter for same color only
        if use_color:
            try:
                same_color_indices = torch.sum(colors_b == current_color, dim=-1)
                distances = distances[0, same_color_indices>0]
                if sum(same_color_indices) == 0: continue
            except Exception as e:
                print(f"ERROR: {e, colors_b.shape, current_color}")
                continue

            
            
        # convert distance to probabilitu
        probs = torch.ones_like(distances) * max(distances.max(), 1.) - distances
        best_prob = torch.max(probs, dim=-1)[0]
        sum_prob = torch.sum(probs, dim=-1)
        if sum_prob == 0: continue
        logMI = -torch.log(best_prob/sum_prob)
        MI.append(logMI)
        
    loss_infoNCE = sum(MI)/len(MI) if len(MI) > 0 else 0

    return loss_infoNCE