import cv2

from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import DataLoader, Sampler
import torch
import tqdm
from torch.nn import functional as F
from src.v2.config import DEVICE, WORKERS
import numpy as np
import pandas as pd
from src.v2.data import AirbusSegmentation, AirbusPseudo
from src.v2.metric import f2, get_iou_vector
from src.utils import save_checkpoint
from tensorboardX import SummaryWriter
from multiprocessing import Pool

from albumentations import (Resize)


def train_iter(inputs):
    model, dataset, optimizer = inputs['model'], inputs['dataset'], inputs['optimizer']
    model.train()
    train_loss = []
    loss_func = model.get_loss()

    for image, mask, has_ship in tqdm.tqdm(
            data.DataLoader(dataset, batch_size=PARAM['batch_size'], shuffle=True, num_workers=WORKERS,
                            pin_memory=True),ascii=True):
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)
        y_pred = model(image)

        optimizer.zero_grad()
        # aux_loss = loss_func(aux, mask).mean()
        mask_loss = loss_func(y_pred, mask).mean()
        loss = mask_loss
        # loss_mask = loss_func(y_pred[:, :1, :, :].contiguous(), mask[:, :1, :, :].contiguous()).mean()
        # loss_border = loss_func(y_pred[:, 1:, :, :].contiguous(), mask[:, 1:, :, :].contiguous()).mean()
        # loss = loss_mask
        loss.backward()

        optimizer.step()
        train_loss.append(mask_loss.item())
        writer.add_scalar('segmentation/batch_loss', mask_loss.item())
        # writer.add_scalar('segmentation/aux_loss', aux_loss.item())

    return np.mean(train_loss)


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[1:3]
    hb, wb = imgb.shape[1:3]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(3, max_height, total_width))
    new_img[:, :ha, :wa] = imga
    new_img[:, :hb, wa:wa + wb] = imgb
    return new_img


def eval(inputs):
    model, dataset = inputs['model'], inputs['dataset']
    model.eval()
    val_loss = []
    mask_ious = []
    border_ious = []
    loss_func = model.get_loss()
    iteration = 0
    for image, mask, has_ship in tqdm.tqdm(
            data.DataLoader(dataset, batch_size=PARAM['eval_batch_size'], shuffle=False, num_workers=WORKERS,
                            pin_memory=True),ascii=True):
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)
        y_pred = model(image)

        # loss_mask = loss_func(y_pred[:, :1, :, :].contiguous(), mask[:, :1, :, :].contiguous()).mean()
        # loss_border = loss_func(y_pred[:, 1:, :, :].contiguous(), mask[:, 1:, :, :].contiguous()).mean()
        loss = loss_func(y_pred, mask).mean()
        # loss = loss_mask + loss_border
        val_loss.append(loss.item())

        mask = mask.cpu().detach().numpy()
        batch_pred = (F.sigmoid(y_pred).cpu().detach().numpy() > 0.5).astype(int)
        mask_iou = get_iou_vector(batch_pred, mask)
        mask_ious.append(mask_iou)

        # mask_iou = get_iou_vector(batch_pred[:, :1, :, :], mask[:, :1, :, :])
        # border_iou = get_iou_vector(batch_pred[:, 1:, :, :], mask[:, 1:, :, :])
        # border_ious.append(border_iou)

    mean_loss = np.mean(val_loss)
    mask_ious = np.mean(mask_ious)
    border_ious = np.mean(border_ious)

    return mean_loss, mask_ious, border_ious


from src.v2.config import segmentator as PARAM

df = pd.read_csv('data/prediction_p.csv')
print("Dataframe size: " + str(df.shape))
df = df[df.ships != 0]
images = df[['ImageId', 'ships']].drop_duplicates()
columns = ['ImageId', 'mask_nb', 'border']
train_df = df[columns]
print("Train size: " + str(train_df.shape))

from src.v2.models.albunet_v2 import load_model

model, state = load_model('last_seg_v2.pth')
writer = SummaryWriter()

optimizer = model.get_optimizer()
train_dataset = AirbusPseudo(train_df, img_size=PARAM['img_size'], mode='train')
val_dataset = AirbusPseudo(train_df, img_size=PARAM['img_size'], mode='val')

for e in range(PARAM['train_epoch']):
    # train_loss = 0
    train_loss = train_iter({
        'model': model,
        'dataset': train_dataset,
        'optimizer': optimizer,
    })
    if e % 4 == 0:
        with torch.no_grad():
            val_loss, mask_iou, border_iou = eval({
                'model': model,
                'dataset': val_dataset,
                'optimizer': optimizer,
            })

    save_checkpoint(model, extra={
        'lb': 0,
        'iou': mask_iou,
        'border_iou': border_iou,
        'val_loss': val_loss,
        'epoch': state['epoch'] + e
    }, checkpoint='last_seg_v2.pth')

    writer.add_scalars('segmentation/losses', {'train_loss': train_loss,
                                               'val_loss': val_loss})
    writer.add_scalars('segmentation/iou', {'border': border_iou, 'mask': mask_iou})

    print("Epoch: %d, Train: %.3f, Val: %.3f, Mask IOU: %.3f, Border IOU: %.3f" % (
        e, train_loss, val_loss, mask_iou, border_iou))
