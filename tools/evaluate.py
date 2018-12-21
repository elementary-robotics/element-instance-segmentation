"""
Please edit "cfg/benchmark.yaml" to specify the necessary parameters for that task.
"""
import argparse
import cv2
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
    "both": "models/ydd.h5",
}


class SDMaskRCNNEvaluator:
    def __init__(self,
                 mode="depth",
                 input_size=512,
                 scaling_factor=2,
                 config_path="cfg/benchmark.yaml"):
        self.element = Element("sd-maskrcnn")
        self.input_size = input_size
        self.scaling_factor = scaling_factor
        self.config_path = config_path
        self.mode = mode
        self.set_mode(b"depth")
        self.element.command_add("segment", self.segment, 5000)
        self.element.command_add("get_mode", self.get_mode, 100)
        self.element.command_add("set_mode", self.set_mode, 5000)
        self.element.command_loop()

    def get_mode(self, _):
        return Response(self.mode)

    def set_mode(self, data):
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

    def scale_and_square(self, img, scaling_factor, size):
        img = cv2.resize(
            img, (int(img.shape[1] / scaling_factor), int(img.shape[0] / scaling_factor)),
            interpolation=cv2.INTER_NEAREST)
        v_pad, h_pad = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
        img = cv2.copyMakeBorder(img, v_pad, v_pad, h_pad, h_pad, cv2.BORDER_REPLICATE)
        return img

    def unscale(self, results, scaling_factor, size):
        masks = results["masks"].astype(np.uint8)
        masks = cv2.resize(
            masks, (int(masks.shape[1] * scaling_factor), int(masks.shape[0] * scaling_factor)),
            interpolation=cv2.INTER_NEAREST)
        v_pad, h_pad = (masks.shape[0] - size[0]) // 2, (masks.shape[1] - size[1]) // 2
        masks = masks[v_pad:-v_pad, h_pad:-h_pad]

        rois = results["rois"] * scaling_factor
        for roi in rois:
            roi[0] -= v_pad
            roi[1] -= h_pad
            roi[2] -= v_pad
            roi[3] -= h_pad
        return masks, rois

    def segment(self, _):
        color_data = self.element.entry_read_n("realsense", "color", 1)
        depth_data = self.element.entry_read_n("realsense", "depth", 1)
        try:
            color_data = color_data[0]["data"]
            depth_data = depth_data[0]["data"]
        except IndexError or KeyError:
            raise Exception("Could not get data. Is the realsense skill running?")

        depth_img = cv2.imdecode(np.frombuffer(depth_data, dtype=np.uint16), -1)
        original_size = depth_img.shape[:2]
        depth_img = self.scale_and_square(depth_img, self.scaling_factor, self.input_size)
        depth_img = self.inpaint(depth_img)
        depth_img = self.normalize(depth_img)

        if self.mode == "both":
            gray_img = cv2.imdecode(np.frombuffer(color_data, dtype=np.uint16), 0)
            gray_img = self.scale_and_square(gray_img, self.scaling_factor, self.input_size)
            input_img = np.zeros((self.input_size, self.input_size, 3))
            input_img[..., 0] = gray_img
            input_img[..., 1] = depth_img
            input_img[..., 2] = depth_img
        else:
            input_img = np.stack((depth_img, ) * 3, axis=-1)

        # Get results and unscale
        results = self.model.detect([input_img], verbose=0)[0]
        masks, rois = self.unscale(results, self.scaling_factor, original_size)

        # Encoded masks in TIF format and package everything in dictionary
        encoded_masks = []
        for mask in masks:
            _, encoded_mask = cv2.imencode(".tif", mask)
            encoded_masks.append(encoded_mask.tobytes())

        response_data = {
            "rois": rois.tolist(),
            "scores": results["scores"].tolist(),
            "masks": encoded_masks
        }
        import matplotlib.pyplot as plt
        from mrcnn import visualize
        color_img = cv2.imdecode(np.frombuffer(color_data, dtype=np.uint16), -1)
        visualize.display_instances(color_img, rois, masks, results['class_ids'], ['bg', 'obj'])
        plt.show()
        return Response(msgpack.packb(response_data, use_bin_type=True))


if __name__ == "__main__":
    evaluator = SDMaskRCNNEvaluator()
