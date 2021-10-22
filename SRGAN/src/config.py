import torch
from torchvision import transforms
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIR = '../data/LR/'
TARGET_DIR = '../data/HR/'
INPUT_DIR_TEST = '../data/LR_test/'
TARGET_DIR_TEST = '../data/HR_test/'
UPSCALE_FACTOR = 4
CROP_SIZE = 88
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
LEARNING_RATE = 0.0002
BATCH_SIZE = 1
NUM_WORKERS = 4
CHANNELS_IMG = 3
NUM_EPOCHS = 150
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
