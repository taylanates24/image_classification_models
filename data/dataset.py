from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import torch
from tqdm import tqdm
import os
import cv2
import numpy as np

def load_image(image_path, image_size):

    image = cv2.imread(image_path)

    height, width = image.shape[:2]
    ratio = image_size / max(height, width)            

    if ratio != 1:

        image = cv2.resize(image, (int(width*ratio), int(height*ratio)), interpolation=cv2.INTER_CUBIC)
    
    return image

def load_image_only(image_path):

    image = cv2.imread(image_path)

    return image



class CustomDataset(Dataset):

    def __init__(self, image_path, annotation_path, image_size=480, augment=False, augmentations=None, letter_box=None) -> None:
        super(CustomDataset, self).__init__()

        self.data_path = image_path
        self.annotation_path = annotation_path

        self.letter_box = letter_box

        self.labels = self.parse_annotations()

        self.image_size = image_size

        self.augment = augment
        self.augmentations = augmentations

        self.image_paths = []

        for img_name in os.listdir(os.path.join(self.data_path)):
            
            self.image_paths.append(os.path.join(self.data_path, img_name))
                
        self.image_paths = sorted(self.image_paths)

        self.transform = transforms.Compose([
            transforms.ToTensor()
            ])


    def __len__(self):

        return len(self.image_paths)
    

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        image_name = image_path.split('/')[-1]
        image = self.load_image_only(image_path=image_path)

        if self.augmentations is not None:
            
            for augment in self.augmentations:
                
                image = augment(image)


        if image.shape[0] != image.shape[1] and self.letter_box:
            
            image = self.letter_box(image=image, size=self.image_size)

        image = self.transform(image.astype('float32')) / 255
        label = self.labels[image_name]
        
        return image, label, image_name


    def load_image(self, image_path):

        image = cv2.imread(image_path)

        height, width = image.shape[:2]
        ratio = self.image_size / max(height, width)            

        if ratio != 1:

            image = cv2.resize(image, (int(width*ratio), int(height*ratio)), interpolation=cv2.INTER_CUBIC)
        
        return image
    
    def load_image_only(self, image_path):

        image = cv2.imread(image_path)

        return image


    def letter_box(self, image, size):

        box = np.full([size, size, image.shape[2]], 127)
        h, w = image.shape[:2]
        h_diff = size - h
        w_diff = size - w
        
        if h_diff > w_diff:
            
            box[int(h_diff/2):int(image.shape[0]+h_diff/2), :image.shape[1], :] = image
 
        else:
            
            box[:image.shape[0], int(w_diff/2):int(image.shape[1]+w_diff/2), :] = image
        
        return box


    def get_statistics(self):

        pass

    
    def parse_annotations(self):

        annotation_file = open(self.annotation_path, "r") 
        annotations = annotation_file.read().split('\n')[:-1]
        labels = {}

        print('Parsing annotations...')

        for ann in tqdm(annotations):
            ann = ann.split(' ')
            labels[ann[0]] = int(ann[1])

        return labels


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


class FPDataset(Dataset):
    def __init__(self, image_path, image_size=64, to_tensor=True, augment=False, augmentations=None, letter_box=False) -> None:
        super(FPDataset, self).__init__()
        self.image_path = image_path
        self.image_size = image_size
        self.augment = augment
        self.augmentations = augmentations
        self.to_tensor = to_tensor

        self.letter_box = letter_box

        self.image_paths = []

        for image_cls in os.listdir(os.path.join(self.image_path)):

            for img_name in os.listdir(os.path.join(self.image_path,image_cls)):
                
                self.image_paths.append(os.path.join(self.image_path,image_cls, img_name))

                
        self.image_paths = sorted(self.image_paths)

        self.transform = transforms.Compose([
            transforms.ToTensor()
            ])

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        image_name = image_path.split('/')[-1]
        image = load_image_only(image_path=image_path)

        if self.augmentations is not None:
            
            for augment in self.augmentations:

                image = augment(image)


        # if image.shape[0] != image.shape[1] and self.letter_box:
            
        #     image = letter_box(image=image, size=self.image_size)

        # if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
        #     image = cv2.resize(image, (self.image_size,self.image_size))
        #image = cv2.resize(image, (self.image_size, self.image_size))
        #image = self.transform(image.astype('float32')) / 255 if self.to_tensor else np.array(image).astype(np.float32) / 255
        #image = self.transform(image)
        label = int(image_name.split('_')[0])
        if label == 9:
            label = 0
        return image, label, image_name


class VideoTypeDataset(Dataset):
    def __init__(self, image_path, image_size=64, to_tensor=True, augment=False, augmentations=None, letter_box=False) -> None:
        super(VideoTypeDataset, self).__init__()
        self.image_path = image_path
        self.image_size = image_size
        self.augment = augment
        self.augmentations = augmentations
        self.to_tensor = to_tensor

        self.letter_box = letter_box

        self.image_paths = []

        for image_cls in os.listdir(os.path.join(self.image_path)):

            for img_name in os.listdir(os.path.join(self.image_path,image_cls)):
                
                self.image_paths.append(os.path.join(self.image_path,image_cls, img_name))

                
        self.image_paths = sorted(self.image_paths)

        self.transform = transforms.Compose([
            transforms.ToTensor()
            ])

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        image_name = image_path.split('/')[-1]
        image = load_image_only(image_path=image_path)

        if self.augmentations is not None:
            
            for augment in self.augmentations:

                image = augment(image, True)

        label_name = image_path.split('/')[1]

        label = 1

        if label_name == 'IR':
            label = 0

            
        return image, label, image_name
    



def load_bbox_crop(image, size, to_tensor=True):

    height, width = image.shape[:2]
    ratio = size / max(height, width)            

    if ratio != 1:

        image = cv2.resize(image, (int(width*ratio), int(height*ratio)), interpolation=cv2.INTER_CUBIC)

    if image.shape[0] != image.shape[1]:
        
        image = letter_box(image=image, size=size)
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size))

    transform = transforms.Compose([
                transforms.ToTensor()
                ])

    image = transform(image.astype('float32')) / 255 if to_tensor else np.array(image).astype(np.float32) / 255

    return image.unsqueeze(0)

def pre_process_image(image_path, image_size):
    
    transform = transforms.Compose([
        transforms.ToTensor()])
    image = load_image(image_path=image_path, image_size=image_size)

    if image.shape[0] != image.shape[1]:
        
        image = letter_box(image=image, size=image_size)
    
    image = transform(image.astype('float32')) / 255
    
    return image.unsqueeze(0)



img = np.random.rand(3,127, 127)
print('asd')
# dataset = CustomDataset(image_path='/workspaces/scene_classification/dataset/images/train',
#                         annotation_path='/workspaces/scene_classification/dataset/annotations/train.txt',
#                         image_size=480,
#                         augment=False)
# model = SceneClassifier()

# for data in dataset:

#     image, label = data
#     image = image.unsqueeze(0)
#     out = model(image)
#     pred = torch.max(model(image), 1)[1]
#     print('asd')