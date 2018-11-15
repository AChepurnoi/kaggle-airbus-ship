import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from PIL import Image
import cv2
import torch
from torch.utils import data
import tqdm

from src.utils import rle_decode, masks_as_image, rle_encode


def combine_masks(values):
    return rle_encode(masks_as_image(values).reshape((768, 768)))


def create_border_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    return mask - erosion


print("Started preprocessing")

df = pd.read_csv('data/prediction.csv')
df['isempty'] = df.apply(lambda x: 1 if x.EncodedPixels is np.nan else 0, axis=1)

ship_counts = df.groupby('ImageId').count()[['EncodedPixels']].rename({'EncodedPixels': 'ships'}, axis=1)

df = df.merge(ship_counts, right_index=True, left_on='ImageId')
df['msize'] = df[df.isempty == 0].apply(lambda x: np.sum(rle_decode(x.EncodedPixels)), axis=1)
groupz = df.groupby('ImageId').groups

k = [(group, df.iloc[groupz[group]].EncodedPixels.values) for group in tqdm.tqdm(groupz)]
kk = [(x[0], combine_masks(x[1])) for x in tqdm.tqdm(k)]
final = pd.DataFrame(kk)
border_mask = final.iloc[:, 1].apply(lambda x: rle_encode(create_border_mask(rle_decode(x))))
final['border'] = border_mask
final.columns = ['ImageId', 'mask_nb', 'border']
final = df[['ImageId', 'ships']].merge(final, left_on='ImageId', right_on='ImageId')
final.to_csv("data/prediction_p.csv", index=False)

print("Finished. Save to data/prediction_p.csv")
