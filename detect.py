import argparse
import os
import time
from pathlib import Path
import torch
import numpy as np
from model import YOLOV5s
from utils.utils import load_model_checkpoint
from utils.plot_utils import cells_to_bboxes, plot_image
from utils.bboxes_utils import non_max_suppression
from PIL import Image
import random
import config
import cv2
import math
def tile_image_tensor(img, tile_size, overlap, save_dir=None):
    """
    Tiles the input image into square tiles of given size and overlap.
    
    Args:
        img (numpy.ndarray): Input image as numpy array (H, W, C).
        tile_size (int): Size of the square tiles (in pixels).
        overlap (int): Overlap between tiles (in pixels).
        save_dir (str): Optional directory to save tiles.

    Returns:
        tiles_tensor (torch.Tensor): Tensor of shape (N, C, H, W) normalized to [0,1]
        offsets (list): list of (x_offset, y_offset) for each tile
    """
    h, w = img.shape[:2]
    tiles = []
    offsets = []

    stride = tile_size - overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            # Handle edge padding if needed
            tile = img[y:y_end, x:x_end]

            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                pad_y = tile_size - tile.shape[0]
                pad_x = tile_size - tile.shape[1]
                tile = cv2.copyMakeBorder(tile, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)
    
            tiles.append(tile)
            
            offsets.append((x, y))

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"tile_x{x}_y{y}.png"
                cv2.imwrite(os.path.join(save_dir, filename), tile)

    # Stack tiles into tensor
    tiles_np = np.stack(tiles, axis=0)  # shape (N, H, W, C)
    tiles_np = tiles_np.transpose((0, 3, 1, 2))  # (N, C, H, W)
    tiles_np = tiles_np.astype(np.float32) / 255.0  # normalize

    tiles_tensor = torch.from_numpy(tiles_np).to(config.DEVICE)

    return tiles_tensor, offsets

if __name__ == "__main__":
    # do not modify
    first_out = config.FIRST_OUT
    nc = len(config.FLIR)
    img_path = "TFront-South-09-31-48-31-04610_jpg.rf.89effbdf6e51b340ad5d12b37e0da7b1.jpg"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,default="model_1" ,help="Indicate the folder inside SAVED_CHECKPOINT")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_8.pth.tar", help="Indicate the ckpt name inside SAVED_CHECKPOINT/model_name")
    parser.add_argument("--img", type=str, default=img_path, help="Indicate path to the img to predict")
    parser.add_argument("--save_pred", action="store_true", help="If save_pred is set, prediction is saved in detections_exp")
    args = parser.parse_args()

    random_img = False

    model = YOLOV5s(first_out=first_out, nc=config.nc, anchors=config.ANCHORS,
                    ch=(first_out * 4, first_out * 8, first_out * 16)).to(config.DEVICE)

    path2model = os.path.join("SAVED_CHECKPOINT", args.model_name, args.checkpoint)
    load_model_checkpoint(model=model, model_name=path2model, last_epoch=100)
    model.eval()
    config.ROOT_DIR = "/".join((config.ROOT_DIR.split("/")[:-1] + ["flir"]))
    # imgs = os.listdir(os.path.join(config.ROOT_DIR, "images", "test"))
    if random_img:
        pass
        # img = np.array(cv2.imread(os.path.join(config.ROOT_DIR, "images", "test", random.choice(imgs)), cv2.IMREAD_UNCHANGED))
    else:

        img = np.array(cv2.imread(args.img, cv2.IMREAD_UNCHANGED))

    
    tiles,_ = tile_image_tensor(img, 640, 150)
    with torch.no_grad():
        out = model(tiles)
    
    bboxes = cells_to_bboxes(out, model.head.anchors, model.head.stride, is_pred=True, to_list=False)
    bboxes = non_max_suppression(bboxes, iou_threshold=0.45, threshold=0.25, tolist=False)
    plot_image(img[0].permute(1, 2, 0).to("cpu"), bboxes, config.FLIR)


