import torch
from models.utils import cosine_distance


def compute_triplet_loss(output_a, output_b, positive_pairs, colors_a, colors_b, n=4, k=0.6):
    pos_distances, neg_distances = [], []
    colors_b = colors_b[0, ...]

    for index in range(0, positive_pairs.shape[1]):
        try:
            index_a = positive_pairs[0, index, 0]
            index_b = positive_pairs[0, index, 1]

            if torch.sum(output_a[:, index_a, :]) == 0.0 or torch.sum(output_b[:, index_b, :]) == 0.0:
                continue

            item_a = output_a[:, index_a, :]
            item_a = item_a.unsqueeze(1).repeat([1, output_b.shape[1], 1])
            current_color = colors_a[0, index_a, :]

            distances = cosine_distance(item_a, output_b)
            assert not torch.isnan(distances).any()
            pos_distances.append(distances[0, index_b])

            same_color_indices = torch.sum(colors_b == current_color, dim=-1)
            same_color_indices = torch.nonzero(same_color_indices, as_tuple=False).squeeze(-1)

            num_anchors = min(n, distances.shape[1] - same_color_indices.shape[0])
            if num_anchors == 0:
                continue
            ignore_pos = torch.zeros_like(distances)
            ignore_pos[0, same_color_indices] = 2.0

            min_distances = torch.topk(distances + ignore_pos, k=num_anchors, largest=False, dim=-1)[0]
            neg_distances.append(min_distances)
        except:
            import pdb; pdb.set_trace()

    pos_distance = torch.mean(torch.stack(pos_distances, dim=-1))
    loss = pos_distance

    if len(neg_distances) > 0:
        neg_distances = torch.cat(neg_distances, dim=-1)
        neg_loss = torch.clamp(torch.full_like(neg_distances, k) - neg_distances, min=0.0)
        neg_loss = torch.mean(neg_loss)
        loss = loss + neg_loss
    return loss


def compute_triplet_three_branches(
        output_a, output_b, x_a, x_b, y_a, y_b,
        positive_pairs, colors_a, colors_b,
):
    loss_out = compute_triplet_loss(output_a, output_b, positive_pairs, colors_a, colors_b)
    loss_x = compute_triplet_loss(x_a, x_b, positive_pairs, colors_a, colors_b)
    loss_y = compute_triplet_loss(y_a, y_b, positive_pairs, colors_a, colors_b)
    loss = loss_out + 0.4 * loss_x + 0.4 * loss_y
    return loss, loss_out, loss_x, loss_y


def compute_triplet_two_branches(
        output_a, output_b, z_a, z_b,
        positive_pairs, colors_a, colors_b,
):
    loss_out = compute_triplet_loss(output_a, output_b, positive_pairs, colors_a, colors_b)
    loss_z = compute_triplet_loss(z_a, z_b, positive_pairs, colors_a, colors_b)
    loss = loss_out + 0.5 * loss_z
    return loss, loss_out, loss_z
