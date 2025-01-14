from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import cv2
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import os
from data.augment import Augmentations
from torchvision.utils import save_image


class CustomDataset(Dataset):
    def __init__(self, image_dir, config_path, phase='train', transform=None):
        """
        Custom dataset for loading images and applying transformations.

        Parameters:
        - image_dir: Path to the directory containing images.
        - config_path: Path to the YAML configuration file.
        - transform: Transformation pipeline (optional, loaded from config if None).
        """
        self.image_dir = image_dir
        self.transform = transform or Augmentations(config_path, phase=phase)

        self.image_paths = []

        for image_cls in os.listdir(os.path.join(image_dir)):

            for img_name in os.listdir(os.path.join(image_dir,image_cls)):
                
                self.image_paths.append(os.path.join(image_dir,image_cls, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = image_path.split('/')[-1]
        image = self.load_image_only(image_path=image_path)

        label_name = image_path.split('/')[1]

        label = 1

        if label_name == 'IR':
            label = 0

        if self.transform:
            image = self.transform(image, label)


        return image, label, image_name

    def load_image_only(self, image_path):

        image = cv2.imread(image_path)

        return image


def letter_box(image, size):
    
    box = np.full([size, size, image.shape[2]], 127)
    h, w = image.shape[:2]
    h_diff = size - h
    w_diff = size - w
    
    if h_diff > w_diff:
        
        box[int(h_diff/2):int(image.shape[0]+h_diff/2), :image.shape[1], :] = image

    else:
        
        box[:image.shape[0], int(w_diff/2):int(image.shape[1]+w_diff/2), :] = image
    
    return box



if __name__ == "__main__":
    config_path = "train.yaml"
    dataset = CustomDataset(image_dir="video_type_dataset", config_path=config_path, phase='train')
    for i in range(20):
        res = dataset.__getitem__(i)
        save_image(res[0][[2, 1, 0], :, :], os.path.join(f'new_out_{i}_{res[2]}.png'))
        print('asd')