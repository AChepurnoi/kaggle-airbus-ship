import cv2

from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import DataLoader, Sampler
import torch
import tqdm
from torch.nn import functional as F
from src.v1.config import DEVICE, WORKERS
import numpy as np
import pandas as pd
from src.v1.data import AirbusSegmentation
from src.v1.metric import f2, get_iou_vector
from src.utils import save_checkpoint
from tensorboardX import SummaryWriter
from multiprocessing import Pool

from albumentations import (Resize)


def train_iter(inputs):
    model, dataset, optimizer = inputs['model'], inputs['dataset'], inputs['optimizer']
    model.train()
    train_loss = []
    loss_func = model.get_loss()

    for image, mask in tqdm.tqdm(
            data.DataLoader(dataset, batch_size=PARAM['batch_size'], shuffle=True, num_workers=WORKERS,
                            pin_memory=True), ascii=True):
        image = image.to(DEVICE)
        y_pred = model(image)

        optimizer.zero_grad()

        loss = loss_func(y_pred, mask.to(DEVICE))
        loss.backward()

        optimizer.step()
        train_loss.append(loss.item())
        writer.add_scalar('segmentation/batch_loss', loss.item())

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


def eval(inputs, should_score):
    model, dataset = inputs['model'], inputs['dataset']
    model.eval()
    val_loss = []
    val_predictions = []
    val_masks = []
    loss_func = model.get_loss()
    iteration = 0
    for image, mask in tqdm.tqdm(
            data.DataLoader(dataset, batch_size=PARAM['eval_batch_size'], shuffle=False, num_workers=WORKERS,
                            pin_memory=True), ascii=True):
        image = image.to(DEVICE)
        y_pred = model(image)

        loss = loss_func(y_pred, mask.to(DEVICE))
        val_loss.append(loss.item())
        batch_pred = F.sigmoid(y_pred).cpu().detach().numpy()
        val_predictions.append(batch_pred)
        val_masks.append(mask)

    mean_loss = np.mean(val_loss)
    val_predictions = np.vstack(val_predictions)
    val_masks = np.vstack(val_masks)

    bin_val_predictions_stacked = (np.array(val_predictions) > 0.5).astype(int)

    if should_score:
        score = f2(val_masks, bin_val_predictions_stacked)
    else:
        score = 0
    iou = get_iou_vector(bin_val_predictions_stacked, val_masks)

    return mean_loss, score, iou


def unnormalize_im(im):
    im = im.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = std * im + mean
    im = np.clip(im, 0, 1)
    im = im.transpose((2, 0, 1))
    return im


from src.v1.config import segmentator as PARAM

df = pd.read_csv('data/processed.csv')
print("Dataframe size: " + str(df.shape))
df = df[df.isempty == 0]
images = df[['ImageId', 'ships']].drop_duplicates()
train, val = train_test_split(images.ImageId.values, test_size=0.05, stratify=images.ships.values, random_state=55)
train_df = df[df.ImageId.isin(train)]
val_df = df[df.ImageId.isin(val)]
print("Train size: " + str(train_df.shape))
print("Val size: " + str(val_df.shape))
from src.v1.models.albunet import load_model
model, state = load_model('last_seg.pth')
writer = SummaryWriter()

for e in range(PARAM['train_epoch']):
    optimizer = model.get_optimizer()
    train_dataset = AirbusSegmentation(train_df, img_size=PARAM['img_size'], mode='train')
    val_dataset = AirbusSegmentation(val_df, img_size=PARAM['img_size'], mode='val')

    # train_loss = 0
    train_loss = train_iter({
        'model': model,
        'dataset': train_dataset,
        'optimizer': optimizer,
    })

    with torch.no_grad():
        val_loss, score, iou = eval({
            'model': model,
            'dataset': val_dataset,
            'optimizer': optimizer,
        }, should_score=False)

    save_checkpoint(model, extra={
        'lb': score,
        'iou': iou,
        'val_loss': val_loss,
        'epoch': state['epoch'] + e
    }, checkpoint='last_seg.pth')

    writer.add_scalars('segmentation/losses', {'train_loss': train_loss,
                                               'val_loss': val_loss})
    if score != 0:
        writer.add_scalar('segmentation/score', score)
    writer.add_scalar('segmentation/iou', iou)

    print("Epoch: %d, Train: %.3f, Val: %.3f, Score: %.3f, IOU: %.3f" % (e, train_loss, val_loss, score, iou))
