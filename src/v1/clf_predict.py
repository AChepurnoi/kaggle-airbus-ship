from torch.utils import data
import torch
import tqdm
from torch.nn import functional as F
from src.v1.config import DEVICE, WORKERS
import numpy as np
import pandas as pd
from src.v1.data import AirbusSegmentation, AirbusClassification
from src.v1.models.albunet import load_model as load_seg_model
from src.v1.models.v1_clf import load_model as load_clf_model

from src.utils import save_checkpoint
from tensorboardX import SummaryWriter
from src.utils import rle_encode

from src.v1.config import evaluation as PARAM


def build_clf_submission(binary_prediction, test_file_list):
    zp_dim = 10
    full_image = np.full((768 - 2 * zp_dim, 768 - 2 * zp_dim), 1)
    full_image = np.pad(full_image, ((zp_dim, zp_dim),), mode='constant', constant_values=0)

    all_masks = []
    for clf_pred in list(binary_prediction):
        if clf_pred[0] == 1:
            p_mask = rle_encode(full_image)
            all_masks.append(p_mask)
        else:
            all_masks.append(' ')

    submit = pd.DataFrame([test_file_list, all_masks]).T
    submit.columns = ['ImageId', 'EncodedPixels']
    return submit


df = pd.read_csv('data/sample_submission_v2.csv')
print("Dataframe size: " + str(df.shape))
clf, clf_state = load_clf_model('best_clf.pth')
# seg, seg_state = load_seg_model('last_seg.pth')
writer = SummaryWriter()

clf.eval()
# seg.eval()
clf_predictions = []

test_dataset = AirbusClassification(df, mode='test')
with torch.no_grad():
    for image in tqdm.tqdm(data.DataLoader(test_dataset, batch_size=PARAM['eval_batch_size'],
                                           shuffle=False, num_workers=WORKERS, pin_memory=True)):
        image = image[0].to(DEVICE)
        y_pred = clf(image)
        clf_predictions.append(F.sigmoid(y_pred).cpu().detach().numpy())

classifier_predictions = (np.vstack(clf_predictions) > 0.5).astype(int)

submit = build_clf_submission(classifier_predictions, df.ImageId.values)
submit.to_csv("classifier_submit.csv", index=False)
