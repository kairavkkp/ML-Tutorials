import numpy as np
import config 
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class MapDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_files = os.listdir(self.input_dir)
        self.target_files = os.listdir(self.target_dir)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index=0):
        input_img_file = self.input_files[index]
        input_img_path = os.path.join(self.input_dir, input_img_file)
        input_image = Image.open(input_img_path)
        input_image = input_image.resize((64 , 64))
        input_image = config.transform(input_image)

        target_img_file = self.target_files[index]
        target_img_path = os.path.join(self.target_dir, target_img_file)
        target_image = Image.open(target_img_path)
        target_image = target_image.resize((256 , 256))
        target_image = config.transform(target_image)

        return input_image, target_image

# if __name__ == "__main__":
#     input_path = config.INPUT_DIR
#     target_path = config.TARGET_DIR
#     dataset = MapDataset(target_path, target_path)
#     loader = DataLoader(dataset, batch_size=5)
    
#     for x, y in loader:
#         print(x, y)