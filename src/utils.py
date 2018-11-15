import torch

import numpy as np
from skimage.measure import label
import cv2
import matplotlib.pyplot as plt
import os

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
from src.v1.config import CHECKPOINT_DIR


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def rle_encode_h(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(np.array(run_lengths).astype(str))


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    runs[::2] -= 1
    return ' '.join(str(x) for x in runs)


def multi_rle_encode(img):
    labels = label(img)
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.int16)
    # if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def mask_overlay(image, mask, color=(0, 1, 0)):
    """
    Helper function to visualize mask on the top of the image
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    weighted_sum = cv2.addWeighted(mask.astype(np.float64), 0.5, image.astype(np.float64), 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img


def imshow(img, mask=None, title=None):
    """Imshow for Tensor."""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    if mask is not None:
        mask = mask.numpy().transpose((1, 2, 0))
        mask = np.clip(mask, 0, 1)
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(mask_overlay(img, mask))
    else:
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def save_checkpoint(model, extra, checkpoint, optimizer=None):
    state = {'state_dict': model.state_dict(),
             'extra': extra}
    if optimizer:
        state['optimizer'] = optimizer.state_dict()

    torch.save(state, CHECKPOINT_DIR + checkpoint)
    print('model saved to %s' % (CHECKPOINT_DIR + checkpoint))


def load_checkpoint(model, checkpoint, optimizer=None):
    exists = os.path.isfile(CHECKPOINT_DIR + checkpoint)
    if exists:
        state = torch.load(CHECKPOINT_DIR + checkpoint)

        model.load_state_dict(state['state_dict'], strict=False)
        optimizer_state = state.get('optimizer')
        if optimizer and optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        print("Checkpoint loaded: %s " % state['extra'])
        return state['extra']
    else:
        print("Checkpoint not found")
        return {'epoch': 0, 'lb_acc': 0}
