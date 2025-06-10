import random 
import numpy as np
import torch
import os
import warnings
import imagesize
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.utils import resize_image
from utils.bboxes_utils import iou_width_height, coco_to_yolo_tensors, non_max_suppression
from utils.plot_utils import plot_image, cells_to_bboxes
import config
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class TiledTrainingDataset(Dataset):
    def __init__(self,
                 root_directory,
                 color_transform=None,
                 spatial_transform=None,
                 tile_size=640,
                 overlap=150,
                 bboxes_format="yolo",
                 train=True,
                 bs=64,
                 ultralytics_loss=True):

        assert bboxes_format == "yolo", "Only YOLO bbox format supported for tiling"

        self.root_directory = root_directory
        self.color_transform = color_transform
        self.spatial_transform = spatial_transform
        self.tile_size = tile_size
        self.overlap = overlap
        self.train = train
        self.bboxes_format = bboxes_format

        self.annot_folder = "train" if train else "val"
        self.img_folder = os.path.join(root_directory, "images", self.annot_folder)
        self.label_folder = os.path.join(root_directory, "labels", self.annot_folder)

        self.tiles_index = []

        image_files = sorted([
            f for f in os.listdir(self.img_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ])
        for img_name in image_files:
            img_path = os.path.join(self.img_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                continue  # skip unreadable or non-4-channel images

            h, w, _ = img.shape
            for y in range(0, h - tile_size + 1, tile_size - overlap):
                for x in range(0, w - tile_size + 1, tile_size - overlap):
                    self.tiles_index.append((img_name, x, y))
        print(f"Total tiles generated: {len(self.tiles_index)}")


    def __len__(self):
        return len(self.tiles_index)

    def __getitem__(self, idx):
        img_name, x_off, y_off = self.tiles_index[idx]
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.label_folder, img_name.rsplit(".", 1)[0] + ".txt")

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Image not found: {img_path}"
        h, w, c = img.shape
        assert c == 4, "Expected 4-channel images"

        tile = img[y_off:y_off + self.tile_size, x_off:x_off + self.tile_size]
        tile = np.ascontiguousarray(tile)

        labels = []
        if os.path.exists(label_path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = np.loadtxt(label_path, delimiter=" ", ndmin=2)
                if labels.size > 0:
                    labels = labels[np.all(labels >= 0, axis=1)]
                    labels = np.atleast_2d(labels)

        abs_labels = []
        for label in labels:
            class_id, x_c, y_c, w_box, h_box = label
            x1 = (x_c - w_box / 2) * w
            y1 = (y_c - h_box / 2) * h
            x2 = (x_c + w_box / 2) * w
            y2 = (y_c + h_box / 2) * h

            # Check if bbox intersects the tile
            if x2 <= x_off or x1 >= x_off + self.tile_size or y2 <= y_off or y1 >= y_off + self.tile_size:
                continue

            # Clip bbox to tile
            x1_tile = np.clip(x1 - x_off, 0, self.tile_size)
            y1_tile = np.clip(y1 - y_off, 0, self.tile_size)
            x2_tile = np.clip(x2 - x_off, 0, self.tile_size)
            y2_tile = np.clip(y2 - y_off, 0, self.tile_size)

            # Convert to YOLO relative coords
            box_w = x2_tile - x1_tile
            box_h = y2_tile - y1_tile
            box_x = x1_tile + box_w / 2
            box_y = y1_tile + box_h / 2

            if box_w <= 0 or box_h <= 0:
                continue

            abs_labels.append([class_id, box_x / self.tile_size, box_y / self.tile_size, box_w / self.tile_size, box_h / self.tile_size])

        labels = np.array(abs_labels)
        if self.spatial_transform:
            aug = self.spatial_transform(image=tile, bboxes=labels)
            tile = aug["image"]
            labels = np.array(aug["bboxes"])

        rgb = tile[:, :, :3]
        alpha = tile[:, :, 3:]

        if self.color_transform:
            aug = self.color_transform(image=rgb, bboxes=labels)
            rgb = aug["image"]
            labels = np.array(aug["bboxes"])

        tile = np.concatenate([rgb, alpha], axis=2) if alpha.shape[2] == 1 else rgb
        tile = tile.transpose((2, 0, 1))
        tile = np.ascontiguousarray(tile)

        labels_tensor = torch.from_numpy(labels) if len(labels) else torch.zeros((0, 5), dtype=torch.float32)

        return torch.from_numpy(tile), labels_tensor


    # this method modifies the target width and height of
    # the images by reshaping them so that the largest size of
    # a given image is set by its closest multiple to 640 (plus some
    # randomness and the other dimension is multiplied by the same scale
    # the purpose is multi_scale training by somehow preserving the
    # original ratio of images

    def adaptive_shape(self, annotations):

        name = "train" if self.train else "val"
        path = os.path.join(
            self.root_directory, "labels",
            "adaptive_ann_{}_{}_br_{}.csv".format(name, self.len_ann, int(self.batch_range))
        )

        if os.path.isfile(path):
            print(f"==> Loading cached annotations for rectangular training on {self.annot_folder}")
            parsed_annot = pd.read_csv(path, index_col=0)
        else:
            print("...Running adaptive_shape for 'rectangular training' on training set...")
            annotations["w_h_ratio"] = annotations.iloc[:, 2] / annotations.iloc[:, 1]
            annotations.sort_values(["w_h_ratio"], ascending=True, inplace=True)

            for i in range(0, len(annotations), self.batch_range):
                size = [annotations.iloc[i, 2], annotations.iloc[i, 1]]  # [width, height]
                max_dim = max(size)
                max_idx = size.index(max_dim)
                size[~max_idx] += 32
                sz = random.randrange(int(self.default_size * 0.9), int(self.default_size * 1.1)) // 32 * 32
                size[~max_idx] = ((sz/size[max_idx])*(size[~max_idx]) // 32) * 32
                size[max_idx] = sz
                if i + self.batch_range <= len(annotations):
                    bs = self.batch_range
                else:
                    bs = len(annotations) - i

                annotations.iloc[i:bs, 2] = size[0]
                annotations.iloc[i:bs, 1] = size[1]

                # sample annotation to avoid having pseudo-equal images in the same batch
                annotations.iloc[i:i+bs, :] = annotations.iloc[i:i+bs, :].sample(frac=1, axis=0)

            parsed_annot = pd.DataFrame(annotations.iloc[:,:3])
            parsed_annot.to_csv(path)

        return parsed_annot

    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)
        return torch.stack(im, 0), label

    @staticmethod
    def collate_fn_ultra(batch):
        images, labels = zip(*batch)  # List[Tensor], List[Tensor]
        batch_size = len(images)

        targets = []
        for i, label in enumerate(labels):
            if label.numel() == 0:
                continue
            label = label.clone()
            img_idx_col = torch.full((label.shape[0], 1), i)
            label = torch.cat([img_idx_col, label], dim=1)  # shape: (N, 6) -> [img_idx, class, x, y, w, h]
            targets.append(label)

        return torch.stack(images, 0), torch.cat(targets, 0) if targets else torch.zeros((0, 6))



class Validation_Dataset(Dataset):
    """COCO 2017 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self,
                 anchors,
                 root_directory=config.ROOT_DIR,
                 transform=None,
                 train=True,
                 S=(8, 16, 32),
                 rect_training=False,
                 default_size=640,
                 bs=64,
                 bboxes_format="yolo",
                 ):
        """
        Parameters:
            train (bool): if true the os.path.join will lead to the train set, otherwise to the val set
            root_directory (path): path to the COCO2017 dataset
            transform: set of Albumentations transformations to be performed with A.Compose
        """
        assert bboxes_format in ["coco", "yolo"], 'bboxes_format must be either "coco" or "yolo"'

        self.batch_range = 64 if bs < 64 else 128
        self.bs = bs
        self.bboxes_format = bboxes_format
        self.transform = transform
        self.S = S
        self.nl = len(anchors[0])
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(self.S).repeat(6, 1).T.reshape(3, 3, 2)
        self.num_anchors = self.anchors.reshape(9,2).shape[0]

        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
        self.rect_training = rect_training
        self.default_size = default_size
        self.root_directory = root_directory
        self.train = train
        self.annot_folder = "train" if train else "val"
        self.fname = f'images/{self.annot_folder}'
        image_folder = os.path.join(self.root_directory, self.fname)
        label_folder = os.path.join(self.root_directory, "labels", self.annot_folder)

        # Collect image files that have matching .txt label files
        self.annotations = []
        for img_name in sorted(os.listdir(image_folder)):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".tif", ".tiff", ".bmp")):
                continue
            label_base = os.path.splitext(img_name)[0]
            label_path = os.path.join(label_folder, label_base + ".txt")
            if os.path.exists(label_path):
                self.annotations.append(img_name)

        self.len_ann = len(self.annotations)

        if self.rect_training:
            self.annotations = self.adaptive_shape(self.annotations)


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_name = self.annotations[idx] if isinstance(self.annotations, list) else self.annotations.iloc[idx, 0]

        tile_size = 640
        overlap = 150
        stride = tile_size - overlap

        tiles = []
        targets_per_tile = []

        # Load full image and labels
        img_path = os.path.join(self.root_directory, self.fname, img_name)
        img = np.array(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
        H, W = img.shape[:2]

        label_base = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.root_directory, "labels", self.annot_folder, label_base + ".txt")
        labels = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
        labels = labels[np.all(labels >= 0, axis=1), :]
        labels[:, 3:5] = np.floor(labels[:, 3:5] * 1000) / 1000

        if self.bboxes_format == "coco":
            labels[:, -1] -= 1
            labels = np.roll(labels, axis=1, shift=1)
            labels[:, 1:] = coco_to_yolo_tensors(labels[:, 1:], w0=W, h0=H)

        # Convert YOLO (x_center, y_center, w, h) to pixel units
        abs_boxes = []
        for label in labels:
            cls, xc, yc, bw, bh = label
            x1 = (xc - bw / 2) * W
            y1 = (yc - bh / 2) * H
            x2 = (xc + bw / 2) * W
            y2 = (yc + bh / 2) * H
            abs_boxes.append([x1, y1, x2, y2, cls])
        abs_boxes = np.array(abs_boxes)

        # Tile the image
        for y0 in range(0, H - tile_size + 1, stride):
            for x0 in range(0, W - tile_size + 1, stride):
                tile = img[y0:y0 + tile_size, x0:x0 + tile_size]
                tile_boxes = []

                for x1, y1, x2, y2, cls in abs_boxes:
                    # Calculate overlap between tile and bbox
                    inter_x1 = max(x1, x0)
                    inter_y1 = max(y1, y0)
                    inter_x2 = min(x2, x0 + tile_size)
                    inter_y2 = min(y2, y0 + tile_size)

                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        # Adjust to local tile coords
                        local_xc = ((inter_x1 + inter_x2) / 2 - x0) / tile_size
                        local_yc = ((inter_y1 + inter_y2) / 2 - y0) / tile_size
                        local_w = (inter_x2 - inter_x1) / tile_size
                        local_h = (inter_y2 - inter_y1) / tile_size
                        tile_boxes.append([cls, local_xc, local_yc, local_w, local_h])

                tile_boxes = np.array(tile_boxes)

                if self.transform:
                    augmentations = self.transform(image=tile,
                                                bboxes=np.roll(tile_boxes, shift=4, axis=1) if len(tile_boxes) else [])
                    tile = augmentations["image"]
                    tile_boxes = np.array(augmentations["bboxes"])
                    if len(tile_boxes):
                        tile_boxes = np.roll(tile_boxes, shift=1, axis=1)

                tile = tile.transpose((2, 0, 1))
                tile = np.ascontiguousarray(tile)

                # Generate YOLO targets for this tile
                classes = tile_boxes[:, 0].tolist() if len(tile_boxes) else []
                bboxes = tile_boxes[:, 1:] if len(tile_boxes) else []
                targets = [torch.zeros((self.num_anchors // 3, tile_size // S, tile_size // S, 6))
                        for S in self.S]

                for idx, box in enumerate(bboxes):
                    iou_anchors = iou_width_height(torch.from_numpy(box[2:4]), self.anchors)
                    anchor_indices = iou_anchors.argsort(descending=True)
                    x, y, width, height = box
                    has_anchor = [False] * 3

                    for anchor_idx in anchor_indices:
                        scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode="floor")
                        anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                        scale_y = targets[scale_idx].shape[1]
                        scale_x = targets[scale_idx].shape[2]
                        i, j = int(scale_y * y), int(scale_x * x)

                        if targets[scale_idx][anchor_on_scale, i, j, 4] == 0 and not has_anchor[scale_idx]:
                            x_cell, y_cell = scale_x * x - j, scale_y * y - i
                            width_cell, height_cell = width * scale_x, height * scale_y
                            targets[scale_idx][anchor_on_scale, i, j, 0:4] = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                            targets[scale_idx][anchor_on_scale, i, j, 4] = 1
                            targets[scale_idx][anchor_on_scale, i, j, 5] = int(classes[idx])
                            has_anchor[scale_idx] = True
                        elif targets[scale_idx][anchor_on_scale, i, j, 4] == 0 and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                            targets[scale_idx][anchor_on_scale, i, j, 4] = -1

                tiles.append(torch.from_numpy(tile))
                targets_per_tile.append(tuple(targets))

        return tiles, targets_per_tile

    # this method modifies the target width and height of
    # the images by reshaping them so that the largest size of
    # a given image is set by its closest multiple to 640 (plus some
    # randomness and the other dimension is multiplied by the same scale
    # the purpose is multi_scale training by somehow preserving the
    # original ratio of images

    def adaptive_shape(self, annotations):

        name = "train" if self.train else "val"
        path = os.path.join(
            self.root_directory, "labels",
            "adaptive_ann_{}_{}_br_{}.csv".format(name, self.len_ann, int(self.batch_range))
        )

        if os.path.isfile(path):
            print(f"==> Loading cached annotations for rectangular training on {self.annot_folder}")
            parsed_annot = pd.read_csv(path, index_col=0)
        else:
            print("...Running adaptive_shape for 'rectangular training' on training set...")
            annotations["w_h_ratio"] = annotations.iloc[:, 2] / annotations.iloc[:, 1]
            annotations.sort_values(["w_h_ratio"], ascending=True, inplace=True)

            for i in range(0, len(annotations), self.batch_range):
                size = [annotations.iloc[i, 2], annotations.iloc[i, 1]]  # [width, height]
                size[0] = size[0] // 32 * 32
                size[1] = size[1] // 32 * 32
                if i + self.batch_range <= len(annotations):
                    bs = self.batch_range
                else:
                    bs = len(annotations) - i

                annotations.iloc[i:bs, 2] = size[0]
                annotations.iloc[i:bs, 1] = size[1]

                # sample annotation to avoid having pseudo-equal images in the same batch
                annotations.iloc[i:i+bs, :] = annotations.iloc[i:i+bs, :].sample(frac=1, axis=0)

            parsed_annot = pd.DataFrame(annotations.iloc[:, :3])
            parsed_annot.to_csv(path)

        return parsed_annot

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == "__main__":

    S = [8, 16, 32]

    anchors = config.ANCHORS

    dataset = Validation_Dataset(anchors=config.ANCHORS,
                                 root_directory=config.ROOT_DIR, transform=None,
                                 train=False, S=S, rect_training=True, default_size=640, bs=4,
                                 bboxes_format="coco")

    # anchors = torch.tensor(anchors)
    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=False)

    for x, y in loader:

        """boxes = cells_to_bboxes(y, anchors, S)[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")"""

        boxes = cells_to_bboxes(y, torch.tensor(anchors), S, to_list=False)
        boxes = non_max_suppression(boxes, iou_threshold=0.6, threshold=0.01, max_detections=300)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes[0])