import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functools import partial
from pycpd import AffineRegistration, DeformableRegistration, RigidRegistration
from matching_components.match_components_by_ucn import MatchingComponents
from components import extract_component_from_image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UCN_object = MatchingComponents(device=device, weight_path='/home/hades/Desktop/prj_toei_colorization/backend/app/ai_core_v2/matching_components/checkpoints/model_010.pth')


def matching_by_cpd(src_points, tg_points):
    def visualize(iteration, error, X, Y, ax):
        plt.cla()
        ax.scatter(X[:, 0], X[:, 1], color='red', label='Target')
        ax.scatter(Y[:, 0], Y[:, 1], color='blue', label='Source')

        plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(iteration, error),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes,
                 fontsize='x-large')

        ax.legend(loc='upper left', fontsize='x-large')
        plt.draw()
        plt.pause(0.001)

    X = np.array(tg_points)
    Y = np.array(src_points)

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    # callback = partial(visualize, ax=fig.axes[0])

    reg = AffineRegistration(**{'X': X, 'Y': Y})
    reg.register(None)
    P = reg.P
    return P


def get_centroid_from_id(id_list, coms):
    points = []
    for id in id_list:
        com = coms[id]
        p = (int(com['centroid'][0]), int(com['centroid'][1]))
        points.append(p)
    return points


def predict_matching(coms1, coms2):
    thres = 0.7
    colors1 = {}
    colors2 = {}
    pairs = {}

    for id1, com in coms1.items():
        color = com['mean_intensity']
        if color not in colors1.keys():
            colors1[color] = [id1]
        else:
            colors1[color].extend([id1])

    for id2, com in coms2.items():
        color = com['mean_intensity']
        if color not in colors2.keys():
            colors2[color] = [id2]
        else:
            colors2[color].extend([id2])

    for color in colors1.keys():
        if color in colors2.keys():
            # Get components in color
            src_id = colors1[color]
            tg_id = colors2[color]

            src_points = get_centroid_from_id(src_id, coms1)
            tg_points = get_centroid_from_id(tg_id, coms2)

            if len(src_points) > 1 and len(tg_points) > 1:
                try:
                    P = matching_by_cpd(src_points, tg_points)
                    for i in range(P.shape[0]):
                        row = P[i, :]
                        max_row = max(row)
                        if max_row > thres:
                            index = np.argmax(row)
                            src_area = coms1[src_id[i]]['area']
                            tg_area = coms2[tg_id[index]]['area']
                            if 0.9 < src_area / tg_area < 1.1:
                                pairs[str(src_id[i]) + '_' + str(tg_id[index])] = {'left': src_id[i], 'right': tg_id[index]}
                except:
                    pass

    return pairs


def ucn_matching(left_img, right_img, left_mask, left_components, right_mask, right_components):
    output_pairs = {}
    # Left is tgt and right is reference
    ucn_pairs, ucn_distance = UCN_object.process(right_img, left_img, right_mask, right_components, left_mask, left_components)

    for tgt_id, ref_id in ucn_pairs.items():
        tgt_id = tgt_id - 1
        ref_id = ref_id[0] - 1
        key = str(tgt_id) + '_' + str(ref_id)
        value = {'left': int(tgt_id), 'right': int(ref_id)}
        output_pairs[key] = value

    return output_pairs





