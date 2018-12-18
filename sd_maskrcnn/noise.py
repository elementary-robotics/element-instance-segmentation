import os
import numpy as np
import skimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

import autolab_core
from perception import DepthImage

base_path = "images/"
out_path = "histograms/"

images = ["depth_0.npy"]
# boxes_list = [[[182, 333, 243, 396], [200, 360, 240, 396], [182, 333, 200, 350], [160, 160, 200, 200], [170, 170, 190, 190], [175, 175, 185, 185], [250, 700, 270, 720]]]


# General squares
# boxes_list = [[[325, 450, 365, 490], [340, 460, 360, 480], [345, 455, 355, 465], [390, 560, 430, 600], [400, 570, 420, 590], [405, 575, 415, 585], [470, 420, 510, 460], [480, 430, 500, 450], [485, 435, 495, 445]]]

# some background squares
boxes_list = [[[330, 800, 350, 820], [550, 800, 570, 820], [400, 150, 420, 170]]]


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def analyze_image_depths(path, bbox, out_name):
    """
    path should lead to a .npy file
    """
    img = np.load(path)
    img = np.reshape(img, img.shape[:2])
    img_slice = img[bbox[0] : bbox[2], bbox[1]: bbox[3]]
    vec = np.ndarray.flatten(img_slice)
    # vec = reject_outliers(vec)

    var = np.var(vec)
    mean = np.mean(vec)
    print("State for {}: Mean: {}, Standard Deviation: {}\n".format(out_name, mean, np.sqrt(var)))

    n, bins, patches = plt.hist(vec, vec.size // 10, facecolor="blue")

    plt.xlabel("depth value")
    plt.ylabel("count")
    plt.title("depth within region")
    plt.grid(True)
    plt.show()

    plt.savefig(os.path.join(out_path, "graph_" + out_name), bbox_inches="tight")
    plt.close()
    depth_img = DepthImage(img_slice)
    depth_img.save(os.path.join(out_path, out_name))


if __name__ == "__main__":
    for img_name in images:
        img_path = os.path.join(base_path, img_name)
        img = np.load(img_path)
        img = np.reshape(img, img.shape[:2])
        DepthImage(img).save(os.path.join(base_path, "depth_0.png"))

    for img_name, boxes in zip(images, boxes_list):
        img_path = os.path.join(base_path, img_name)
        for box in boxes:
            out_name = "img_{}_box_{}_{}_{}_{}.png" \
                       .format(img_name, box[0], box[1], box[2], box[3])
            analyze_image_depths(img_path, box, out_name)