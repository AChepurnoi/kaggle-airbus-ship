from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import DataLoader, Sampler
import torch
import tqdm
from torch.nn import functional as F
from src.v1.config import DEVICE, WORKERS
import numpy as np
import pandas as pd
from src.v1.data import AirbusSegmentation, AirbusClassification
from src.v1.models.v1_clf import load_model
from src.utils import save_checkpoint
from sklearn.metrics import classification_report, accuracy_score
from tensorboardX import SummaryWriter


def train_iter(inputs):
    model, dataset, optimizer = inputs['model'], inputs['dataset'], inputs['optimizer']
    writer = inputs['writer']
    model.train()
    train_loss = []
    loss_func = model.get_loss()
    for image, has_ship in tqdm.tqdm(
            data.DataLoader(dataset, batch_size=PARAM['batch_size'], shuffle=True, num_workers=WORKERS,
                            pin_memory=True)):
        image = image.to(DEVICE)
        y_pred = model(image)

        loss = loss_func(y_pred, has_ship.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss.append(loss.item())
        writer.add_scalar('classifier/batch_loss', loss.item())

    return np.mean(train_loss)


def eval(inputs):
    model, dataset = inputs['model'], inputs['dataset']
    model.eval()
    val_loss = []
    predictions = []
    gts = []
    loss_func = model.get_loss()
    for image, has_ship in tqdm.tqdm(
            data.DataLoader(dataset, batch_size=PARAM['batch_size'], shuffle=False, num_workers=WORKERS,
                            pin_memory=True)):
        image = image.to(DEVICE)
        y_pred = model(image)

        loss = loss_func(y_pred, has_ship.to(DEVICE))

        val_loss.append(loss.item())

        predictions.append(F.sigmoid(y_pred).cpu().detach().numpy())
        gts.append(has_ship)

    mean_loss = np.mean(val_loss)
    predictions = np.vstack(predictions)
    gts = np.vstack(gts)

    bin_predictions = (np.array(predictions) > 0.5).astype(int)
    report = classification_report(gts, bin_predictions)
    acc = accuracy_score(gts, bin_predictions)
    return mean_loss, acc, report

from src.config import classifier as PARAM

df = pd.read_csv('data/processed.csv')
images = df[['ImageId', 'isempty']].drop_duplicates()
train, val = train_test_split(images.ImageId.values, test_size=0.05, stratify=images.isempty.values, random_state=55)
train_df = df[df.ImageId.isin(train)]
val_df = df[df.ImageId.isin(val)]
print("Val size: " + str(val_df.shape))
model, state = load_model('last_clf.pth')
writer = SummaryWriter()

for e in range(PARAM['train_epoch']):
    accuracy = []

    optimizer = model.get_optimizer()
    train_dataset = AirbusClassification(train_df, mode='train')
    val_dataset = AirbusClassification(val_df, mode='val')

    train_loss = train_iter({
        'model': model,
        'dataset': train_dataset,
        'optimizer': optimizer,
        'writer': writer
    })

    with torch.no_grad():
        val_loss, acc, report = eval({
            'model': model,
            'dataset': val_dataset,
            'optimizer': optimizer,
        })
    accuracy.append(acc)

    save_checkpoint(model, extra={
        'lb': acc,
        'epoch': state['epoch'] + e
    }, checkpoint='last_clf.pth')

    writer.add_scalars('classifier/losses', {'train_loss': train_loss,
                                             'val_loss': val_loss})
    writer.add_scalar('classifier/metric', acc)

    print("Epoch: %d, Train: %.3f, Val: %.3f, Acc: %.3f" % (e, train_loss, val_loss, acc))
    print(report)
