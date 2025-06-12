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
from torch.utils.data._utils.collate import default_collate

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

        abs_labels = np.array(abs_labels)

        if abs_labels.shape[0] > 0:
            class_labels = abs_labels[:, 0].astype(int).tolist()
            bboxes = abs_labels[:, 1:].tolist()
        else:
            class_labels = []
            bboxes = []

        # Spatial transform
        if self.spatial_transform and bboxes:
            aug = self.spatial_transform(image=tile, bboxes=bboxes, class_labels=abs_labels[:, 0].astype(int).tolist())
            tile = aug["image"]
            bboxes = aug["bboxes"]
            class_labels = aug["class_labels"]

            abs_labels = []
            for cls, bbox in zip(class_labels, bboxes):
                abs_labels.append([cls] + list(bbox))
            abs_labels = np.array(abs_labels)
        else:
            abs_labels = np.array([[cls] + list(bbox) for cls, bbox in zip(class_labels, bboxes)]) if bboxes else np.array([])

        # Split channels for color transform
        rgb = tile[:, :, :3]
        alpha = tile[:, :, 3:]

        # Color transform (apply only on rgb)
        if self.color_transform and abs_labels.shape[0] > 0:
            aug = self.color_transform(image=rgb, bboxes=abs_labels[:, 1:].tolist(), class_labels=abs_labels[:, 0].astype(int).tolist())
            rgb = aug["image"]
            bboxes = aug["bboxes"]
            class_labels = aug["class_labels"]

            abs_labels = []
            for cls, bbox in zip(class_labels, bboxes):
                abs_labels.append([cls] + list(bbox))
            abs_labels = np.array(abs_labels)

        # Recombine channels
        tile = np.concatenate([rgb, alpha], axis=2) if alpha.shape[2] == 1 else rgb
        tile = tile.transpose((2, 0, 1))
        tile = np.ascontiguousarray(tile)

        labels_tensor = torch.from_numpy(abs_labels) if abs_labels.size else torch.zeros((0, 5), dtype=torch.float32)

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


def encode_yolo_targets(labels, anchors, strides, num_classes, img_size):
    """
    Args:
        labels: tensor of shape [N, 5], each row [class_id, x_center, y_center, w, h], normalized (0-1)
        anchors: list of 3 tensors, each tensor shape [num_anchors_per_scale, 2] (width, height normalized by stride)
        strides: list of 3 ints, stride per scale (e.g., [8,16,32])
        num_classes: int, number of classes
        img_size: int, input image size (assumed square)

    Returns:
        targets: list of 3 tensors, each shape [batch=1, grid, grid, num_anchors, 5+num_classes]
    """

    targets = []
    device = labels.device if isinstance(labels, torch.Tensor) else 'cpu'

    for scale_idx, stride in enumerate(strides):
        grid_size = img_size // stride
        num_anchors = anchors[scale_idx].shape[0]

        # Initialize target tensor with zeros
        target_tensor = torch.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), device=device)

        if labels.shape[0] == 0:
            targets.append(target_tensor)
            continue

        # Scale boxes to grid size
        boxes_scaled = labels[:, 1:5] * grid_size  # x_c, y_c, w, h scaled to grid

        class_ids = labels[:, 0].long()

        for i, (cls, box) in enumerate(zip(class_ids, boxes_scaled)):
            x_c, y_c, w, h = box

            # Find the best anchor (IoU with anchors) for this scale
            box_wh = torch.tensor([w, h], device=device)
            anchor_shapes = anchors[scale_idx].to(device)

            # Calculate IoU of box_wh with each anchor at this scale (simple IoU)
            inter_area = torch.min(anchor_shapes[:, 0], box_wh[0]) * torch.min(anchor_shapes[:, 1], box_wh[1])
            union_area = (anchor_shapes[:, 0] * anchor_shapes[:, 1]) + (box_wh[0] * box_wh[1]) - inter_area
            ious = inter_area / union_area

            best_anchor_idx = torch.argmax(ious)

            # Get grid cell indices
            grid_x, grid_y = int(x_c), int(y_c)

            if grid_x >= grid_size or grid_y >= grid_size:
                continue  # ignore boxes outside the grid (should rarely happen)

            # Set objectness = 1
            target_tensor[grid_y, grid_x, best_anchor_idx, 0] = 1.0
            # Set box coordinates (relative to cell)
            target_tensor[grid_y, grid_x, best_anchor_idx, 1:5] = torch.tensor([x_c - grid_x, y_c - grid_y, w, h], device=device)
            # Set class one-hot vector
            target_tensor[grid_y, grid_x, best_anchor_idx, 5 + cls] = 1.0

        targets.append(target_tensor)

    return targets

class Validation_Dataset(Dataset):
    def __init__(self,
                 anchors,
                 root_directory,
                 transform=None,
                 train=True,
                 S=(8, 16, 32),
                 rect_training=False,
                 default_size=640,
                 bs=64,
                 bboxes_format="yolo"):
        
        assert bboxes_format == "yolo", "Only YOLO bbox format supported for tiling"
        assert not train, "This is a validation dataset, set train=False"

        self.anchors = anchors
        self.root_directory = root_directory
        self.transform = transform  # only color transforms, applied to RGB only
        self.S = S
        self.rect_training = rect_training
        self.tile_size = default_size
        self.bs = bs
        self.bboxes_format = bboxes_format
        self.overlap = 150

        self.annot_folder = "val"
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
                continue

            h, w, _ = img.shape
            for y in range(0, h - self.tile_size + 1, self.tile_size - self.overlap):
                for x in range(0, w - self.tile_size + 1, self.tile_size - self.overlap):
                    self.tiles_index.append((img_name, x, y))

        print(f"Total validation tiles generated: {len(self.tiles_index)}")

    def __len__(self):
        return len(self.tiles_index)

    def __getitem__(self, idx):
        img_name, x_off, y_off = self.tiles_index[idx]
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.label_folder, img_name.rsplit(".", 1)[0] + ".txt")

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
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

            # Skip boxes outside the tile
            if x2 <= x_off or x1 >= x_off + self.tile_size or y2 <= y_off or y1 >= y_off + self.tile_size:
                continue

            x1_tile = np.clip(x1 - x_off, 0, self.tile_size)
            y1_tile = np.clip(y1 - y_off, 0, self.tile_size)
            x2_tile = np.clip(x2 - x_off, 0, self.tile_size)
            y2_tile = np.clip(y2 - y_off, 0, self.tile_size)

            box_w = x2_tile - x1_tile
            box_h = y2_tile - y1_tile
            box_x = x1_tile + box_w / 2
            box_y = y1_tile + box_h / 2

            if box_w <= 0 or box_h <= 0:
                continue

            abs_labels.append([
                class_id,
                box_x / self.tile_size,
                box_y / self.tile_size,
                box_w / self.tile_size,
                box_h / self.tile_size
            ])

        abs_labels = np.array(abs_labels)

        # --- Spatial Transform (none for val) ---

        # Split channels for color transform
        rgb = tile[:, :, :3]
        alpha = tile[:, :, 3:]

        # Apply optional color transform (on rgb only)
        if self.transform and abs_labels.shape[0] > 0:
            aug = self.transform(
                image=rgb,
                bboxes=abs_labels[:, 1:].tolist(),
                class_labels=abs_labels[:, 0].astype(int).tolist()
            )
            rgb = aug["image"]
            bboxes = aug["bboxes"]
            class_labels = aug["class_labels"]
            abs_labels = np.array([[cls] + list(bbox) for cls, bbox in zip(class_labels, bboxes)])

        # Recombine channels
        tile = np.concatenate([rgb, alpha], axis=2) if alpha.shape[2] == 1 else rgb
        tile = tile.transpose((2, 0, 1)).astype(np.float32)
        tile_tensor = torch.from_numpy(tile)

        # Convert abs_labels to tensor
        if abs_labels.shape[0] > 0:
            abs_labels_tensor = torch.from_numpy(abs_labels).float()
        else:
            abs_labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
        anchors_normalized = [
            torch.tensor(anchors, dtype=torch.float32) / stride
            for anchors, stride in zip(self.anchors, self.S)
        ]

        # Now encode targets for all scales using encode_yolo_targets
        targets = encode_yolo_targets(
            abs_labels_tensor,
            anchors=anchors_normalized,  # list of tensors [[num_anchors, 2], ...] normalized by stride
            strides=self.S,  # list of ints like [8,16,32]
            num_classes=config.nc,
            img_size=self.tile_size
        )

        return tile_tensor, targets

    @staticmethod
    def collate_fn(batch):
        imgs, targets = list(zip(*batch))
        imgs = torch.stack(imgs, dim=0)
        
        targets_per_scale = list(zip(*targets))
        targets_stacked = [torch.stack(scale_targets, dim=0) for scale_targets in targets_per_scale]

        return imgs, targets_stacked




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