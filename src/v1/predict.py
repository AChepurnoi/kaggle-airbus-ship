from torch.utils import data
from torch.utils.data import DataLoader, Sampler
import torch
import tqdm
from torch.nn import functional as F
from src.v1.config import DEVICE
import numpy as np
import pandas as pd

from src.v1.data import AirbusSegmentation
from src.v1.models.v1 import load_model
from src.utils import rle_encode


def test(inputs):
    model, dataset = inputs['model'], inputs['dataset']
    model.eval()
    all_predictions = []
    for image in tqdm.tqdm(data.DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=True)):
        image = image[0].type(torch.float).to(DEVICE)
        y_pred = torch.sigmoid(model(image)).cpu().detach().numpy()
        all_predictions.append(y_pred)
    return all_predictions


def build_submission(binary_prediction, test_file_list):
    all_masks = []
    for p_mask in list(binary_prediction):
        p_mask = rle_encode(p_mask)
        all_masks.append(' '.join(map(str, p_mask)))
    submit = pd.DataFrame([test_file_list, all_masks]).T
    submit.columns = ['ImageId', 'EncodedPixels']
    return submit


sample_sub = pd.read_csv('data/sample_submission_v2.csv')
test_dataset = AirbusSegmentation(sample_sub, mode='test')

model, state = load_model('last.pth')

stacked_predictions = test({
    'model': model,
    'dataset': test_dataset
})

stacked_predictions = np.vstack(stacked_predictions)
binary_prediction = (stacked_predictions > 0.5).astype(int)

submit = build_submission(binary_prediction, sample_sub.ImageId.values)

submit.to_csv("submit.csv", index=False)
