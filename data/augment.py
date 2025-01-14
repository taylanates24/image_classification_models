import random
from PIL import Image
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import adjust_sharpness
import os
from torchvision.transforms import InterpolationMode
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import torch
import random

class RandomCompose:
    def __init__(self, always_apply_first, always_apply_last, random_transforms, aug_prob, use_all=False, k=None, center_crop_ratio=None, center_crop_type='center',
                 corner_crop_ratio=None):
        self.always_apply_first = always_apply_first
        self.always_apply_last = always_apply_last
        self.random_transforms = random_transforms
        self.aug_prob = aug_prob
        self.use_all = use_all
        self.k = k
        self.center_crop_ratio = center_crop_ratio
        self.crop_type = center_crop_type
        self.corner_crop_ratio = corner_crop_ratio

    def __call__(self, img):

        for t in self.always_apply_first:
            img = t(img)

        if self.center_crop_ratio is not None:
            if self.crop_type == 'upper-left':
                img = self.centercrop(img, self.center_crop_ratio, 'center')
                img = self.centercrop(img, self.corner_crop_ratio, self.crop_type)
            else:
                img = self.centercrop(img, self.center_crop_ratio, 'center')
        
        if random.uniform(0, 1) < self.aug_prob:
            if self.k is None:
                self.k = random.randint(0, len(self.random_transforms)) if not self.use_all else len(self.random_transforms)

            selected_transforms = random.sample(self.random_transforms, self.k)

            for t in selected_transforms:
                img = t(img)

        for t in self.always_apply_last:
            img = t(img)

        return img
    
    def centercrop(self, image, ratio, corner="center"):
        """
        Crop the center or upper-left corner of a PIL image based on the given ratio.

        Args:
            image (PIL.Image.Image): The input image to crop.
            ratio (float): The ratio of the cropped image size relative to the original size (0 < ratio <= 1).
            corner (str): The corner to crop from ("center" or "upper-left").

        Returns:
            PIL.Image.Image: The cropped image.
        """
        if not (0 < ratio <= 1):
            raise ValueError("Ratio must be between 0 and 1.")

        # Get the original image dimensions
        width, height = image.size

        # Calculate the dimensions of the cropped area
        crop_width = int(width * ratio)
        crop_height = int(height * ratio)

        if corner == "center":
            # Calculate the cropping box coordinates for center crop
            left = (width - crop_width) // 2
            top = (height - crop_height) // 2
        elif corner == "upper-left":
            # Set the cropping box coordinates for upper-left crop
            left = 0
            top = 0
        else:
            raise ValueError("Invalid corner option. Use 'center' or 'upper-left'.")

        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))

        return cropped_image
    
class LetterBoxTransform:
    def __init__(self, size, fill_value=127, interpolation=Image.Resampling.BICUBIC):
        self.size = size
        self.fill_value = fill_value
        self.interpolation = interpolation
    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("Input image should be a PIL Image.")
        
        w, h = image.size
        # Resize the image so that the longer side matches the desired size
        if w > h:
            new_w = self.size
            new_h = int(h * (self.size / w))
        else:
            new_h = self.size
            new_w = int(w * (self.size / h))
        
        resized_image = image.resize((new_w, new_h), interpolation=self.interpolation)
        
        # Create a new image with the desired size and fill value
        box = Image.new('RGB', (self.size, self.size), (self.fill_value, self.fill_value, self.fill_value))
        
        # Paste the resized image onto the center of the new image
        paste_x = (self.size - new_w) // 2
        paste_y = (self.size - new_h) // 2
        box.paste(resized_image, (paste_x, paste_y))
        
        return box
    

class Augmentations:

    def __init__(self, opt, img_size=None, phase='train', k=None) -> None:

        self.random_idx = list(range(100))
        self.aug_prob = opt.get('aug_prob', 0.5)
        self.img_size = img_size

        interpolation = InterpolationMode.BICUBIC
        self.letterbox = opt.get('letterbox', False)
        
        self.always_first = []
        self.always_last = []

        self.transforms_list = []

        self.always_first.append(transforms.ToPILImage())


        perspective = opt.get('perspective')
        perspective = perspective[:2] if perspective[-1] else 0

        if perspective:
            self.transforms_list.append(transforms.RandomPerspective(distortion_scale=perspective[0], p=perspective[1], 
                                                                     interpolation=interpolation))
            
        sharpness = opt.get('sharpness')
        self.sharpness = sharpness[:2] if sharpness[-1] else 0

        scale = opt.get('scale')
        scale = scale[:2] if scale[-1] else [1,1]


        translate = opt.get('translate')
        translate = translate[:2] if translate[-1] else [0,0]


        rotate = opt.get('rotate')
        rotate = rotate[:2] if rotate[-1] else 0


        shear = opt.get('shear')
        shear = shear[:2] if shear[-1] else 0
        affine_prob = opt.get('affine_prob', 0.2)
        if scale != [1,1] or translate != [0,0] or rotate or shear:
            self.transforms_list.append(transforms.RandomApply(torch.nn.ModuleList([
                                                    transforms.RandomAffine(degrees=rotate, 
                                                                            translate=translate, 
                                                                            scale=scale, shear=shear,
                                                                            interpolation=interpolation)
                                                                            ]),
                                                                            p=affine_prob))

        # Color Augmentations

        brightness = opt.get('brightness') 
        brightness = brightness[:2] if brightness[-1] else 0
            
        contrast = opt.get('contrast')
        contrast = contrast[:2] if contrast[-1] else 0

        saturation = opt.get('saturation')
        saturation = saturation[:2] if saturation[-1] else 0

        hue = opt.get('hue')
        hue = hue[:2] if hue[-1] else 0

        jitter_prob = opt.get('jitter_prob', 0.2)

        if brightness or contrast or saturation or hue or jitter_prob:
            self.transforms_list.append(transforms.RandomApply(torch.nn.ModuleList([
                transforms.ColorJitter(brightness=brightness, 
                                        contrast=contrast, 
                                        saturation=saturation, 
                                        hue=hue)
                                        ]),
                                        p=jitter_prob))

        # Horizontal Flip
        fliplr = opt.get('fliplr')
        if fliplr and fliplr[-1]:
            self.always_last.append(transforms.RandomHorizontalFlip(p=fliplr[0]))
        # Crop
        random_crop = opt.get('random_crop', None)
        
        if random_crop:
            size_crop, scale_crop, crop_prob = random_crop

            if crop_prob:
                # self.transforms_list.append(transforms.RandomApply(torch.nn.ModuleList([
                #         transforms.RandomResizedCrop(size=size_crop, scale=scale_crop, interpolation=interpolation)]),
                #         p=crop_prob))
                self.always_first.append(transforms.RandomResizedCrop(size=size_crop, scale=scale_crop, interpolation=interpolation))
        # Center Crop
        center_crop = opt.get('center_crop')
        center_crop_val = opt.get('center_crop_val')
        corner_crop_ratio = None
        if center_crop:
            center_crop_ratio = center_crop[0] if center_crop[1] else None
            center_crop_type =  None
            

        elif center_crop_val:
            center_crop_ratio = center_crop_val[0] if center_crop_val[1] else None
            corner_crop_ratio = center_crop_val[1] if center_crop_val[1] else None
            center_crop_type = center_crop_val[-1]


        resize = opt.get('resize')
        if resize and resize[-1]:
            height = resize[0] if img_size is None else img_size
            width = resize[1] if img_size is None else img_size
            self.always_last.append(transforms.Resize((height, width), interpolation=interpolation))


        if self.letterbox:
            self.always_last.append(LetterBoxTransform(size=self.img_size, fill_value=127))
        self.always_last.append(transforms.ToTensor())
        self.transforms = RandomCompose(always_apply_first=self.always_first, always_apply_last=self.always_last,
                                        random_transforms=self.transforms_list, aug_prob=self.aug_prob,
                                        k=k, center_crop_ratio=center_crop_ratio, center_crop_type=center_crop_type,
                                        corner_crop_ratio=corner_crop_ratio)


    def __call__(self, image, save=False, image_name='augmented_image'):

        if save:
            import cv2
            idx = self.random_idx[0]
            if idx == 0:
                print('asd')
            del self.random_idx[0]
            cv2.imwrite(os.path.join(f'or_{idx}_{image_name}.png'), image)
        image = self.transforms(image)
        # Save the augmented image
        if self.sharpness:
            if random.uniform(0, 1) <= self.sharpness[1]:
                sh_fac = self.sharpness[0]
                sharpness_factor = random.uniform(sh_fac[0], sh_fac[1])
                image = adjust_sharpness(image, sharpness_factor=sharpness_factor)
        if save:
            #idx = self.random_idx[0]
            #del self.random_idx[0]
            save_image(image[[2, 1, 0], :, :], os.path.join(f'out_{idx}_{image_name}.png'))
                
        return image



# # Example usage
# if __name__ == "__main__":
#     opt = {
#         'aug_prob': 1,
#         'letterbox': False,
#         'resize': [160, 160, 1], # [-5, 5]
#         'perspective': [0.2, 0.5, 1],
#         'scale': [0.8, 1.2, 1],
#         'translate': [0.1, 0.1, 0],
#         'rotate': [-30, 30, 1],
#         'shear': [-10, 10, 1],
#         'sharpness': [[0.8, 2], 1, 1],
#         'brightness': [0.8, 1.2, 1],
#         'contrast': [0.8, 1.3, 1],
#         'saturation': [0.7, 1.3, 1],
#         'hue': [0.1, 0.1, 0],
#         'add_grayscale': [0.2, 0],
#         'fliplr': [0.5, 1],
#         'crop': [0.1, 0.1, 0],
#         'center_crop': [0.8, 0.8, 0],
#         'some_of': 3
#     }
    
#     opt2 = {
#         'aug_prob': 0,
#         'letterbox': False,
#         'resize': [160, 160, 1], # [-5, 5]
#         'perspective': [0.2, 0.5, 1],
#         'scale': [0.8, 1.2, 1],
#         'translate': [0.1, 0.1, 0],
#         'rotate': [-30, 30, 1],
#         'shear': [-10, 10, 1],
#         'sharpness': [[0.8, 2], 1, 1],
#         'brightness': [0.8, 1.2, 0],
#         'contrast': [0.8, 1.3, 0],
#         'saturation': [0.7, 1.3, 0],
#         'hue': [0.1, 0.1, 0],
#         'add_grayscale': [0.2, 0],
#         'fliplr': [0.5, 1],
#         'crop': [0.1, 0.1, 0],
#         'center_crop': [0.8, 0.8, 0],
#         'some_of': 3
#     }

#     img_size = 160
#     aug = Augmentations(opt, img_size)

#     # Load an example image
#     import cv2
#     image = cv2.imread('example_image.png')  # Replace with your image path

#     # Apply augmentations and save the image
#     aug(image, save=True, image_name='example_image_aug2')