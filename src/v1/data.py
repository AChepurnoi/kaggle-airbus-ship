import torch
from torch.utils import data
import numpy as np
import os
from PIL import Image
import random
import cv2

from src.v1.config import DEBUG, TRAIN_PATH, TEST_PATH
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


class AirbusSegmentation(data.Dataset):

    def __init__(self, data, img_size=384, aug=True, mode='train'):
        self.data = data
        self.mode = mode
        if mode is 'train':
            self.images = data.ImageId.unique()
            self._aug = Compose([
                Flip(),
                RandomRotate90(),
                ShiftScaleRotate(),
                Normalize(),
                # Resize(256, 256),
                Resize(img_size, img_size),

            ])
        elif mode is 'test' or mode is 'val':
            self.images = data.ImageId.unique()
            self._aug = Compose([
                Normalize(),
                # Resize(256, 256),
                Resize(img_size, img_size),

                # PadIfNeeded(768, 768)
            ])
        else:
            raise RuntimeError()

    def __getitem__(self, i):
        image_id = self.images[i]
        if self.mode is 'train' or self.mode is 'val':
            im_path = os.path.join(TRAIN_PATH, image_id)
            image = cv2.imread(im_path)
            mask = masks_as_image(self.data[self.data.ImageId == image_id].EncodedPixels.values)
            image, mask = self.apply_aug(image, mask)
            image = torch.from_numpy(image).float().permute([2, 0, 1])
            mask = torch.from_numpy(mask).float().unsqueeze(2).permute([2, 0, 1])
            # mask = torch.from_numpy(mask).float().permute([2, 0, 1])

            return image, mask
        elif self.mode is 'test':
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
            image, mask = augmented["image"], augmented["mask"]
        else:
            image = self._aug(image=image)['image']

        return image, mask

    def __len__(self):
        if DEBUG:
            return 2
        else:
            return len(self.images)


class AirbusClassification(data.Dataset):

    def __init__(self, df, aug=True, mode='train'):
        self.data = df
        self.mode = mode
        if mode is 'train':
            self.images = df.ImageId.unique()
            self._aug = Compose([
                Normalize(),
                Flip(),
                Resize(224, 224)
            ])
        elif mode is 'test' or mode is 'val':
            self.images = df.ImageId.unique()
            self._aug = Compose([
                Normalize(),
                Resize(224, 224)
            ])
        else:
            raise RuntimeError()

    def __getitem__(self, i):
        image_id = self.images[i]

        if self.mode is 'train' or self.mode is 'val':
            im_path = os.path.join(TRAIN_PATH, image_id)
            image = cv2.imread(im_path)
            has_ship = not np.all(self.data[self.data.ImageId == image_id].isempty.values)
            image, _ = self.apply_aug(image)
            image = torch.from_numpy(image).float().permute([2, 0, 1])
            return image, torch.FloatTensor([has_ship])
        elif self.mode is 'test':
            im_path = os.path.join(TEST_PATH, image_id)
            image = cv2.imread(im_path)
            image, _ = self.apply_aug(image)
            image = torch.from_numpy(image).float().permute([2, 0, 1])
            return (image,)
        else:
            raise RuntimeError()

    def apply_aug(self, image, mask=None):
        if mask is not None:
            data = {"image": image, "mask": mask}
            augmented = self._aug(**data)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image = self._aug(image=image)['image']

        return image, mask

    def __len__(self):
        if DEBUG:
            return 2
        else:
            return len(self.images)
