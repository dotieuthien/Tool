import os
import cv2
from PIL import Image, ImageDraw
import numpy as np
import skimage.measure as measure
import imageio


def show(mask, out=None, seed=0):
    if seed: np.random.seed(seed) 
    color = np.random.randint(1, 255, (mask.max()+1, 3)).astype(np.uint8)
    component_mask = color[mask]
    component_mask[mask == 0] = (0,0,0)

    img = Image.fromarray(component_mask)

    #TODO: adding debug
    db_img = ImageDraw.Draw(img)
    r = 5
    h, w = mask.shape[:2]
    for i in range(mask.max() + 1):
        ys, xs = np.where(mask == i)
        if len(xs) == 0: continue
        x, y = xs[len(xs) // 2], ys[len(xs) // 2]
        db_img.ellipse((max(x-r, 0), max(y-r,0), min(x+r, w), min(y+r,h)), fill = 'blue')
        db_img.text((x+r, y+r), text=f"{i}", fill='black')

    if out is None:
        img.show()
    else:
        img.save(os.path.basename(out).split(".")[0] + ".png")

def get_mask(comps, im_size):
    mask = np.zeros(shape=im_size[:2], dtype='int')
    for id, comp in enumerate(comps):
        mask[comps[id]['coords'][:, 0], comps[id]['coords'][:, 1]] = id + 1
    
    return mask

def load_comps_from_mask(mask):
    # Pre-processing image
    labels = measure.label(mask, connectivity=1, background=0)
    regions = measure.regionprops(labels, intensity_image=mask)
    index = 0
    components = dict()
    for region in regions:
        # import pdb; pdb.set_trace()
        components[index] = {
            "centroid": np.array(region.centroid),
            "area": region.area,
            "image": region.image.astype(np.uint8) * 255,
            "label": region.label,
            "coords": region.coords,
            "bbox": region.bbox,
            "min_intensity": region.min_intensity,
            "mean_intensity": region.mean_intensity,
            "max_intensity": region.max_intensity,
        }
        mask[region.coords[:, 0], region.coords[:, 1]] = index + 1
        index += 1
    
    components = [components[i] for i in range(0, len(components))]
    return components

if __name__ == '__main__':
    from rules.component_wrapper import ComponentWrapper
    component_wrapper = ComponentWrapper()

    color_image = "/home/cain/data/toei/toei_easy_training_set/05_ggg05_s20_038_S_T_A_CHM_NON/color/0001.tga"
    sketch = "/home/cain/data/toei/toei_easy_training_set/05_ggg05_s20_038_S_T_A_CHM_NON/sketch/0001.tga"
    method = ComponentWrapper.EXTRACT_COLOR

    color_image = cv2.cvtColor(np.array(Image.open(color_image).convert("RGB")), cv2.COLOR_RGB2BGR)
    sketch = cv2.cvtColor(np.array(Image.open(sketch).convert("RGB")), cv2.COLOR_RGB2BGR)
    mask, components = component_wrapper.process(color_image, sketch, method)
    # recover_comps = load_comps_from_mask(mask)
    # recover_mask = get_mask(recover_comps, im_size=color_image.shape)
    # show(mask, out='original_mask.png', seed=1)
    # show(recover_mask, out='recover_mask.png', seed=1)

    # test = cv2.imread('original_mask.png', 'L')
    # print(test.shape)
    # print(test)
    mask[mask==0] = 512
    print(mask.max())

    #test save
    imageio.imwrite("mask_16.PNG", mask.astype(np.uint16))

    #test read
    img = imageio.imread("mask_16.PNG")
    print(img.max())
