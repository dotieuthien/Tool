from models.residual_models.pyramid_net import UNet
import torch
from torch import nn
from models.utils import count_parameters, get_coord_features
from copy import deepcopy

# BORROW FROM SUPERGLUE
def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0):
        for layer, name in zip(self.layers, self.names):
            src0 = desc0
            delta0 = layer(desc0, src0)
            desc0 = (desc0 + delta0)
        return desc0


class SingleChannelBlock(nn.Module):
    def __init__(self, feature_dim, layers=[]):
        super().__init__()
        # self.downblock = nn.Conv2d(feature_dim, feature_dim, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), dilation=(0,1,1))

    def forward(self, x):
        # out = self.downblock(x)
        out = torch.cat([x, get_coord_features(x)], dim=1)
        return out

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        return self.encoder(kpts)


################################################################
class UCN_Transformer(nn.Module):
    """SuperGlue feature matching middle-end
    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold
    The correspondence ids use -1 to indicate non-matching points.
    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763
    """
    default_config = {
        'descriptor_dim': 256,
        # 'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, base_weight_path=None, dropout=0.0):
        super().__init__()
        self.config = {**self.default_config}

        #TODO: UCN set up
        self.ucn = UNet(base_weight_path, dropout)

        #TODO: location + shape desc
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        #TODO: transformer-based attention setup
        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        #TODO: final projection to feature space
        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        # bin_score = torch.nn.Parameter(torch.tensor(1.))
        # self.register_parameter('bin_score', bin_score)

        # assert self.config['weights'] in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        # self.load_state_dict(torch.load(str(path)))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config['weights']))

    def forward(self, image, labels, kpts=None):
        #TODO: run UCN
        desc0 = self.ucn(image, labels)
        desc0 = desc0.transpose(1,2)

        #TODO: get key points
        if kpts is not None:
            kpts0 = kpts.transpose(1,2)
            # Keypoint normalization.
            # kpts0 = normalize_keypoints(kpts0, image.shape)

            # Keypoint MLP encoder.
            desc0 = desc0 + self.kenc(kpts0)

        # Multi-layer Transformer network.
        desc0 = self.gnn(desc0)

        # Final MLP projection.
        mdesc0 = self.final_proj(desc0)
        mdesc0 = mdesc0.transpose(1,2)
        return mdesc0

from PIL import Image
from rules.component_wrapper import ComponentWrapper, resize_mask_and_fix_components
import cv2
from loader.features_utils import convert_to_stacked_mask
import numpy as np

def main():
    import time
    torch.cuda.benchmark = True

    paths = [
        "/mnt/lustre/home/cain/code/github/dev/hades_painting/0001.tga",
    ]
    sketch_image = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    color_image = cv2.cvtColor(np.array(Image.open(paths[0]).convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = color_image.shape[:2]

    # extract components
    component_wrapper = ComponentWrapper()
    input_label, input_components = component_wrapper.process(color_image, None, ComponentWrapper.EXTRACT_COLOR)
    print(len(input_components))
    input_label, input_components = resize_mask_and_fix_components(input_label, input_components, (768, 512))
    print(len(input_components), input_label.shape, input_label.dtype, input_label.min(), input_label.max())
    input_label = convert_to_stacked_mask(input_label).astype(np.int32)
    input_label = torch.from_numpy(input_label).float().unsqueeze(0)

    model = UNet(None, 0.0)
    model.eval()
    print(count_parameters(model))
    start = time.time()
    output = model(torch.zeros([1, 3, 512, 768]), input_label)
    print(output.shape, torch.mean(output))
    print(time.time() - start)


    model = UCN_Transformer(None, 0.0)
    model.eval()
    print(count_parameters(model))

    kpts = torch.from_numpy(np.array([np.concatenate([comp['centroid'], [len(comp['coords'])]]) for comp in input_components]))
    kpts[:,0] = kpts[:,0] / h
    kpts[:,1] = kpts[:,1] / w
    kpts[:,2] = kpts[:,2] / h / w
    kpts = kpts.float().unsqueeze(0)
    start = time.time()
    output = model(torch.zeros([1, 3, 512, 768]), input_label, kpts)
    print(output.shape, torch.mean(output))
    print(time.time() - start)


if __name__ == "__main__":
    main()
