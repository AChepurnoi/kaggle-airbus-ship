import torch
from torch.utils import data
import numpy as np
import os
from PIL import Image
import random
import cv2
from src.utils import rle_decode
from src.v2.config import DEBUG, TRAIN_PATH, TEST_PATH
from src.utils import masks_as_image
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    ToGray,
    RandomCrop,
    OpticalDistortion,
    RandomRotate90,
    RandomSizedCrop,
    Transpose,
    GridDistortion,
    Blur,
    OneOf,
    InvertImg,
    GaussNoise,
    OneOf,
    ElasticTransform,
    MedianBlur,
    ShiftScaleRotate,
    Rotate,
    Normalize,
    Crop,
    CLAHE,
    Flip,
    LongestMaxSize,
    RandomScale,
    PadIfNeeded,
    Compose,
    RandomBrightness,
    RandomContrast,
    convert_bboxes_to_albumentations,
    filter_bboxes_by_visibility,
    denormalize_bbox,
    RandomGamma)


def create_border_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    res = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    return res


class AirbusSegmentation(data.Dataset):

    def __init__(self, data, img_size=384, aug=True, mode='train'):
        self.data = data
        self.mode = mode
        if mode is 'train':
            self.images = data.ImageId.unique()
            self._aug = Compose([
                HorizontalFlip(),
                VerticalFlip(),
                OneOf([
                    RandomRotate90(),
                    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0),
                ], p=1),
                OneOf([
                    CLAHE(),
                    RandomBrightness(),
                    RandomContrast(),
                    RandomGamma()
                ], p=1),
                Normalize(),
                RandomCrop(img_size, img_size)
                # Resize(256, 256),
                # Resize(img_size, img_size),

            ])
        elif mode is 'test' or mode is 'val':
            self.images = data.ImageId.unique()
            self._aug = Compose([
                Normalize(),
                # Resize(256, 256),
                # Resize(img_size, img_size),
                # RandomCrop(img_size, img_size)
                # PadIfNeeded(768, 768)
            ])
        else:
            raise RuntimeError()

    def __getitem__(self, i):
        if self.mode is 'train' or self.mode is 'val':
            image_id, mask, border = self.data.iloc[i]
            im_path = os.path.join(TRAIN_PATH, image_id)
            image = cv2.imread(im_path)
            mask = rle_decode(mask)
            # border = create_border_mask(mask)
            # mask = np.dstack((mask, border))
            image, mask = self.apply_aug(image, mask)
            image = torch.from_numpy(image).float().permute([2, 0, 1])
            mask = torch.from_numpy(mask).float().unsqueeze(2).permute([2, 0, 1])
            has_ship = torch.sum(mask) > 0
            # mask = torch.from_numpy(mask).float().permute([2, 0, 1])

            return image, mask, has_ship
        elif self.mode is 'test':
            image_id = self.images[i]
            im_path = os.path.join(TEST_PATH, image_id)
            image = cv2.imread(im_path)
            image, _ = self.apply_aug(image)
            image = torch.from_numpy(image).float().permute([2, 0, 1])
            return (image, image_id)
        else:
            raise RuntimeError()

    def apply_aug(self, image, mask=None):
        if mask is not None:
            data = {"image": image, "mask": mask}
            augmented = self._aug(**data)
            a_image, a_mask = augmented["image"], augmented["mask"]
            if mask.sum() < 1:
                return self.apply_aug(image, mask)
        else:
            a_image = self._aug(image=image)['image']
            a_mask = None
        return a_image, a_mask

    def __len__(self):
        if DEBUG:
            return 8
        else:
            return len(self.data)


class AirbusPseudo(data.Dataset):

    def __init__(self, data, img_size=384, mode='train'):
        self.data = data
        self.mode = mode
        if mode is 'train':
            self.images = {im_id: cv2.imread(os.path.join(TEST_PATH, im_id)) for im_id in data.ImageId.unique()}
            self._aug = Compose([
                HorizontalFlip(),
                VerticalFlip(),
                OneOf([
                    RandomRotate90(),
                    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2),
                ], p=1),
                OneOf([
                    CLAHE(),
                    RandomBrightness(),
                    RandomContrast(),
                    RandomGamma()
                ], p=1),
                Normalize(),
                RandomCrop(img_size, img_size)
                # Resize(256, 256),
                # Resize(img_size, img_size),

            ])
        elif mode is 'test' or mode is 'val':
            self.images = {im_id: cv2.imread(os.path.join(TEST_PATH, im_id)) for im_id in data.ImageId.unique()}
            self._aug = Compose([
                Normalize(),
                # Resize(256, 256),
                # Resize(img_size, img_size),
                # RandomCrop(img_size, img_size)
                # PadIfNeeded(768, 768)
            ])
        else:
            raise RuntimeError()

    def __getitem__(self, i):
        if self.mode is 'train' or self.mode is 'val':
            image_id, mask, border = self.data.iloc[i]
            # im_path = os.path.join(TEST_PATH, image_id)
            # image = cv2.imread(im_path)
            image = self.images[image_id]
            mask = rle_decode(mask)
            image, mask = self.apply_aug(image, mask)
            image = torch.from_numpy(image).float().permute([2, 0, 1])
            mask = torch.from_numpy(mask).float().unsqueeze(2).permute([2, 0, 1])
            has_ship = torch.sum(mask) > 0

            return image, mask, has_ship
        elif self.mode is 'test':
            image_id = self.images[i]
            im_path = os.path.join(TEST_PATH, image_id)
            image = cv2.imread(im_path)
            image, _ = self.apply_aug(image)
            image = torch.from_numpy(image).float().permute([2, 0, 1])
            return (image, image_id)
        else:
            raise RuntimeError()

    def apply_aug(self, image, mask=None):
        if mask is not None:
            data = {"image": image, "mask": mask}
            augmented = self._aug(**data)
            a_image, a_mask = augmented["image"], augmented["mask"]
            if mask.sum() < 1:
                return self.apply_aug(image, mask)
        else:
            a_image = self._aug(image=image)['image']
            a_mask = None
        return a_image, a_mask

    def __len__(self):
        if DEBUG:
            return 8
        else:
            return len(self.data)
