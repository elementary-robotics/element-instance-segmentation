"""
Please edit "cfg/benchmark.yaml" to specify the necessary parameters for that task.
"""
import argparse
import cv2
import matplotlib.pyplot as plt
import msgpack
import numpy as np
import os
from atom import Element
from atom.messages import LogLevel, Response
from autolab_core import YamlConfig
from mrcnn import model as modellib
from resize import scale_to_square
from sd_maskrcnn.config import MaskConfig

MODES = set(["depth", "both"])
MODEL_PATHS = {
    "depth": "models/sd_maskrcnn.h5",
    "both": "models/sd_maskrcnn.h5",
}


class SDMaskRCNNEvaluator:
    def __init__(self, config_path="cfg/benchmark.yaml"):
        self.element = Element("sd-maskrcnn")
        self.config_path = config_path
        self.select_mode(b"depth")
        self.element.command_add("segment", self.segment, 5000)
        self.element.command_add("select_mode", self.select_mode, 5000)
        self.element.command_loop()

    def select_mode(self, data):
        mode = data.decode().strip().lower()
        if mode not in MODES:
            return Response(f"Invalid mode {mode}")
        self.mode = mode
        config = YamlConfig(self.config_path)
        inference_config = MaskConfig(config['model']['settings'])
        inference_config.GPU_COUNT = 1
        inference_config.IMAGES_PER_GPU = 1

        model_path = MODEL_PATHS[self.mode]
        model_dir, _ = os.path.split(model_path)
        self.model = modellib.MaskRCNN(
            mode=config['model']['mode'], config=inference_config, model_dir=model_dir)
        self.model.load_weights(model_path, by_name=True)
        self.element.log(LogLevel.INFO, f"Loaded weights from {model_path}")
        return Response(f"Mode switched to {self.mode}")

    def inpaint(self, img, missing_value=0):
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

    def normalize(self, img, max_dist=1000):
        img = np.clip(img / max_dist, 0, 1) * 255
        img = np.clip(img + (255 - img.max()), 0, 255)
        return img.astype(np.uint8)

    def segment(self, _):
        color_data = self.element.entry_read_n("realsense", "color", 1)
        depth_data = self.element.entry_read_n("realsense", "depth", 1)
        try:
            color_data = color_data[0]["data"]
            depth_data = depth_data[0]["data"]
        except IndexError or KeyError:
            raise Exception("Could not get data. Is the realsense skill running?")

        depth_img = cv2.imdecode(np.frombuffer(depth_data, dtype=np.uint16), -1)
        h, w = depth_img.shape
        depth_img = cv2.resize(
            depth_img, (int(w / 2.5), int(h / 2.5)), interpolation=cv2.INTER_NEAREST)
        # depth_img = scale_to_square(depth_img)
        v_pad, h_pad = (512 - depth_img.shape[0]) // 2, (512 - depth_img.shape[1]) // 2
        depth_img = cv2.copyMakeBorder(
            depth_img, v_pad, v_pad, h_pad, h_pad, cv2.BORDER_CONSTANT, value=0)
        depth_img = self.inpaint(depth_img)
        depth_img = self.normalize(depth_img)
        depth_img = np.stack((depth_img, ) * 3, axis=-1)

        """
        color_img = cv2.imdecode(np.frombuffer(color_data, dtype=np.uint16), -1)[:, :, ::-1]
        color_img = scale_to_square(color_img)
        v_pad, h_pad = (512 - color_img.shape[0]) // 2, (512 - color_img.shape[1]) // 2
        color_img = cv2.copyMakeBorder(
            color_img, v_pad, v_pad, h_pad, h_pad, cv2.BORDER_CONSTANT, value=0)
        """

        results = self.model.detect([depth_img], verbose=0)[0]

        masks = []
        for i in range(results["masks"].shape[-1]):
            _, mask = cv2.imencode(".tif", results["masks"][..., i].astype(np.uint8))
            masks.append(mask.tobytes())

        response_data = {
            "rois": results["rois"].tolist(),
            "scores": results["scores"].tolist(),
            "masks": masks
        }
        return Response(msgpack.packb(response_data, use_bin_type=True))


if __name__ == "__main__":
    evaluator = SDMaskRCNNEvaluator()
