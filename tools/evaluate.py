"""
Please edit "cfg/benchmark.yaml" to specify the necessary parameters for that task.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import cv2
from resize import scale_to_square

from autolab_core import YamlConfig
from sd_maskrcnn.config import MaskConfig
from mrcnn import model as modellib, visualize


def detect(model, image):
    results = model.detect([image], verbose=0)
    r = results[0]
    
    # Save info
    r_info = {
        'rois': r['rois'],
        'scores': r['scores'],
        'class_ids': r['class_ids']
    }
    print(r_info)
    print(r["masks"].shape)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ['bg', 'obj'])
    plt.savefig("masked.png")
    plt.show()


def inpaint(img, missing_value=0):
    # cv2 inpainting doesn't handle the border properly
    # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale
    return img


def gradients(img):
    # depth = cv2.GaussianBlur(depth, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_DEFAULT)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return DepthImage(grad_x), DepthImage(grad_y), DepthImage(grad)


def normalise(img):
    img = np.clip(img / 1000, 0, 1) * 255
    img = np.clip(img + (255 - img.max()), 0, 255)
    return img.astype(np.uint8)


if __name__ == "__main__":
    from atom import Element
    element = Element("sd-maskrcnn")

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Benchmark SD Mask RCNN model")
    conf_parser.add_argument("--config", action="store", default="cfg/benchmark.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)


    inference_config = MaskConfig(config['model']['settings'])
    inference_config.GPU_COUNT = 1
    inference_config.IMAGES_PER_GPU = 1
    model_dir, _ = os.path.split(config['model']['path'])
    model = modellib.MaskRCNN(mode=config['model']['mode'], config=inference_config, model_dir=model_dir)

    # Load trained weights
    print("Loading weights from ", config['model']['path'])
    model.load_weights(config['model']['path'], by_name=True)


    color_data = element.entry_read_n("realsense", "color", 1)
    depth_data = element.entry_read_n("realsense", "depth", 1)
    try:
        color_data = color_data[0]["data"]
        depth_data = depth_data[0]["data"]
    except IndexError or KeyError:
        raise Exception("Could not get data. Is the realsense skill running?")

    depth_img = cv2.imdecode(np.frombuffer(depth_data, dtype=np.uint16), -1)
    h, w = depth_img.shape
    depth_img = cv2.resize(depth_img, (int(w / 2.5), int(h / 2.5)), interpolation=cv2.INTER_NEAREST)
    # depth_img = scale_to_square(depth_img)
    v_pad, h_pad = (512 - depth_img.shape[0]) // 2, (512 - depth_img.shape[1]) // 2
    depth_img = cv2.copyMakeBorder(depth_img, v_pad, v_pad, h_pad, h_pad, cv2.BORDER_CONSTANT, value=0)
    depth_img = inpaint(depth_img)
    depth_img = normalise(depth_img)
    depth_img = np.stack((depth_img,)*3, axis=-1)

    color_img = cv2.imdecode(np.frombuffer(color_data, dtype=np.uint16), -1)[:,:,::-1]
    color_img = scale_to_square(color_img)
    v_pad, h_pad = (512 - color_img.shape[0]) // 2, (512 - color_img.shape[1]) // 2
    color_img = cv2.copyMakeBorder(color_img, v_pad, v_pad, h_pad, h_pad, cv2.BORDER_CONSTANT, value=0)

    # cv2.imshow("img", color_img[:,:,::-1])
    # cv2.waitKey(5)
    detect(model, depth_img)
