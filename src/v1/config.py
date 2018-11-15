import os
import torch

CHECKPOINT_DIR = 'chk/'
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIRECTORY = 'data'
DEBUG = False
TRAIN_PATH = 'data/train_v2/'
TEST_PATH = 'data/test_v2/'

LOAD_CHECKPOINT = True
WORKERS = 0 if DEBUG else 16

segmentator = {
    'encoder_lr': 1e-5,
    'lr': 1e-5,
    'L2': 0.0001,
    'batch_size': 4,
    'img_size': 768,
    # 'img_size': 384,
    'eval_batch_size': 4,
    'train_epoch': 300
}

classifier = {
    'lr': 1e-4,
    'batch_size': 24,
    'train_epoch': 3
}

evaluation = {
    'eval_batch_size': 2,
    # 'img_size': 384,
    'img_size': 768

}

# PARAM = classifier
