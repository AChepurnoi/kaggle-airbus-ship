from scipy import ndimage
from tensorboardX import SummaryWriter
from torch.utils import data
import torch
import tqdm
from src.v2.config import DEVICE, WORKERS
import numpy as np
import pandas as pd
import cv2
from src.v2.data import AirbusSegmentation
from src.v2.models.albunet_v2 import load_model
from src.utils import rle_encode, rle_encode_h
from src.v2.config import evaluation as PARAM


def build_submission(binary_prediction, test_file_list):
    all_masks = []
    for p_mask in list(binary_prediction):
        p_mask = rle_encode(p_mask)
        all_masks.append(p_mask)
    submit = pd.DataFrame([test_file_list, all_masks]).T
    submit.columns = ['ImageId', 'EncodedPixels']
    return submit


def split_mask(mask):
    threshold = 0.5
    threshold_obj = 30  # ignor predictions composed of "threshold_obj" pixels or less
    labled, n_objs = ndimage.label(mask > threshold)
    result = []
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        if (obj.sum() > threshold_obj): result.append(obj)
    return result


clf_submit = pd.read_csv('classifier_submit.csv')
# clf_submit = pd.read_csv('kernel_sub.csv')
non_empty = clf_submit[~clf_submit['EncodedPixels'].isna()]
empty_images = clf_submit[clf_submit['EncodedPixels'].isna()]

test_dataset = AirbusSegmentation(non_empty, img_size=PARAM['img_size'], mode='test')

model, state = load_model('last_seg_v2.pth')
ship_list_dict = []
writer = SummaryWriter()
iteration = 0

flips = [
    [2],
    [3],
    [2, 3]
]


def rotate_tensor(tensor, k):
    if k == 0:
        return tensor
    n_t = tensor.numpy()
    axes = (len(tensor.shape) - 2, len(tensor.shape) - 1)
    n_t = np.rot90(n_t, k=k, axes=axes).copy()
    return torch.FloatTensor(n_t)


with torch.no_grad():
    model.eval()
    for image, image_ids in tqdm.tqdm(data.DataLoader(test_dataset, batch_size=PARAM['eval_batch_size'],
                                                      num_workers=WORKERS, pin_memory=True)):
        image = image.type(torch.float).to(DEVICE)

        y_pred_src = model(image).cpu().detach().numpy()
        y_pred = [y_pred_src]
        for flip in flips:
            y_pred_aug = model(image.flip(flip)).flip(flip).cpu().detach().numpy()
            y_pred.append(y_pred_aug)

        y_pred = np.array(y_pred)
        y_pred = torch.sigmoid(torch.from_numpy(np.stack(y_pred).mean(axis=0)))
        y_pred = y_pred.numpy()

        for i in range(image.shape[0]):
            p = y_pred[i]
            id = image_ids[i]
            img_size = PARAM['img_size']
            # if img_size != 768:
                # p = cv2.resize(p.reshape((img_size, img_size, 1)), (768, 768))

            masks = split_mask(p)
            if (len(masks) == 0):
                ship_list_dict.append({'ImageId': id, 'EncodedPixels': np.nan})
            for mask in masks:
                ship_list_dict.append({'ImageId': id, 'EncodedPixels': rle_encode_h(mask)})

segmentation = pd.DataFrame(ship_list_dict)
prediction = pd.concat([empty_images, segmentation], sort=False).sort_values(by='ImageId')

prediction.to_csv("submission%.3f.csv" % state['iou'], index=False)
