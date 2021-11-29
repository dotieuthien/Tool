import numpy as np
import torch
from labelme.matching_components.match_components_by_rules import *
from labelme.matching_components.shallow_net import UNet
from labelme.matching_components.pyramid_net import UNet as PNet


class MatchingComponents:
    def __init__(self, device, weight_path):
        self.image_size = (768, 512)
        self.mean = [2.0, 7.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]
        self.std = [0.8, 2.0, 10.0, 10.0, 30.0, 20.0, 30.0, 1.0]

        self.mean = np.array(self.mean)[:, np.newaxis][:, np.newaxis]
        self.std = np.array(self.std)[:, np.newaxis][:, np.newaxis]

        self.device = device
        self.model = PNet(None, 0.0)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device)["model"])
        self.model.to(self.device)
        self.model.eval()

        self.pairs = None
        self.distances = None

    def process(self,
                ref_sketch_img,
                tgt_sketch_img,
                ref_mask,
                ref_components,
                tgt_mask,
                tgt_components):
        self.pairs = dict()

        new_tgt_components = [tgt_components[i] for i in range(0, len(tgt_components))]
        new_ref_components = [ref_components[i] for i in range(0, len(ref_components))]

        max_tgt_label = max([c["label"] for c in tgt_components.values()]) if len(tgt_components) else 0
        max_ref_label = max([c["label"] for c in ref_components.values()]) if len(ref_components) else 0
        self.distances = np.full([max_tgt_label + 1, max_ref_label + 1], np.inf)

        # TODO: Ensure that when no components can be extracted, the sketch automatically becomes the output.
        if len(new_tgt_components) == 0 or len(new_ref_components) == 0:
            return self.pairs, self.distances

        # Get data
        ref_list, tgt_list = get_input_data(
            ref_sketch_img,
            tgt_sketch_img,
            ref_mask,
            new_ref_components,
            tgt_mask,
            new_tgt_components,
            self.image_size, self.mean, self.std)

        ref_features, ref_mask, new_ref_components, ref_graph, ref_mapping = ref_list
        tgt_features, tgt_mask, new_tgt_components, tgt_graph, tgt_mapping = tgt_list

        # Run the model
        with torch.no_grad():
            ref_features = ref_features.float().to(self.device)
            ref_mask = ref_mask.to(self.device)
            tgt_features = tgt_features.float().to(self.device)
            tgt_mask = tgt_mask.to(self.device)

            ref_latent = self.model(ref_features, ref_mask)
            tgt_latent = self.model(tgt_features, tgt_mask)
            if tgt_latent is None or ref_latent is None:
                return self.pairs, self.distances

        all_pairs, best_pairs, distances = get_top_distances_of_target_component(
            ref_latent, tgt_latent,
            ref_graph, tgt_graph,
            new_ref_components, new_tgt_components)

        for tgt_index, ref_indices in all_pairs.items():
            mapped_tgt_index = tgt_mapping[tgt_index]
            best_ref_index = best_pairs[tgt_index][0]
            self.pairs[mapped_tgt_index] = [ref_mapping[best_ref_index]]

            for ref_index in ref_indices:
                mapped_ref_index = ref_mapping[ref_index]
                self.distances[mapped_tgt_index, mapped_ref_index] = distances[tgt_index, ref_index]

        return self.pairs, self.distances
