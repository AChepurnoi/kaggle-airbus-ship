import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from PIL import Image
import cv2
import torch
from torch.utils import data

from src.utils import rle_decode
print("Started preprocessing")

df = pd.read_csv('data/train_ship_segmentations_v2.csv')
df['isempty'] = df.apply(lambda x: 1 if x.EncodedPixels is np.nan else 0, axis=1)

ship_counts = df.groupby('ImageId').count()[['EncodedPixels']].rename({'EncodedPixels': 'ships'}, axis=1)

df = df.merge(ship_counts, right_index=True, left_on='ImageId')
df['msize'] = df[df.isempty == 0].apply(lambda x: np.sum(rle_decode(x.EncodedPixels)), axis=1)
df.to_csv('data/processed.csv', index=False)
print("Finished. Save to data/processed.csv")
