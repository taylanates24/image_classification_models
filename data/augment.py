import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode
from data.utils import crop_fn_pil as crop_fn
from PIL import Image
import yaml
import random
import sys

sys.path.append('./')

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
    def __init__(self, config, img_size=None, phase='train'):
        """
        Initialize augmentations based on a configuration file.

        Parameters:
        - config_path: Path to the YAML configuration file.
        - img_size: Target image size for resizing (optional).
        - phase: 'train' or 'val', determines augmentations to apply.
        """
        if type(config) == str:

            with open(config, 'r') as f:
                config = yaml.safe_load(f)

        self.phase = phase
        self.img_size = img_size
        self.aug_prob = config.get('aug_prob', 0.5)

        interpolation = InterpolationMode.BICUBIC
        self.transforms_list = []
        self.always_first = []
        self.always_last = []

        # Always apply to PIL image first
        self.always_first.append(transforms.ToPILImage())
        self.center_crop = config['dataset'][phase].get('center_crop', None)

        if self.center_crop:

            self.center_crop_ratio = self.center_crop['ratio']
            self.center_crop_type = 'center'

        self.crop = config['dataset'][phase].get('crop', None)

        if self.crop:

            self.crop_ratio = self.crop['ratio']
            self.crop_type = self.crop['type']

        self.random_invert = None


        if phase == 'train':

            if config['dataset'][phase]['random_invert'].get('apply', False):

                ri_prob = config['dataset'][phase]['random_invert']['p']
                self.random_invert = transforms.RandomInvert(ri_prob)
            # Load augmentations dynamically
            for aug in config['dataset']['augmentations']:

                aug_type = aug['type']
                params = aug.get('parameters', {})
                prob = aug.get('probability', 1.0)

                aug_class = getattr(transforms, aug_type, None)

                if aug_class:

                    transform = aug_class(**params)

                    if prob < 1.0:

                        self.transforms_list.append(transforms.RandomApply([transform], p=prob))

                    else:

                        self.transforms_list.append(transform)

                else:

                    raise ValueError(f"Unsupported augmentation: {aug_type}")

        # Resize for validation or as the final step
        resize = config['dataset'][phase].get('img_size', img_size)

        if resize:

            height = width = resize
            self.always_last.append(transforms.Resize((height, width), interpolation=interpolation))

        # Always apply to Tensor as the last step
        self.always_last.append(transforms.ToTensor())

        self.transforms = None

        if len(self.transforms_list):
            self.transforms = transforms.Compose(self.transforms_list)
            
        self.always_first = transforms.Compose(self.always_first)
        self.always_last = transforms.Compose(self.always_last)

    def __call__(self, image, label):
        """Apply the composed transformations to the image."""

        if self.always_first:

            image = self.always_first(image)

        if label == 0 and self.random_invert is not None:

            image = self.random_invert(image)
        if self.transforms is not None:
            image = self.transforms(image)

        if self.center_crop:

            image = crop_fn(image, self.center_crop_ratio, self.center_crop_type)

        if self.crop:

            crop_type = "upper-left"

            if self.crop_type == 'random':

                crop_type = random.choice(['upper-left', 'upper-right', 'lower-left', 'lower-right'])
            image = crop_fn(image, self.crop_ratio, crop_type)

        if self.always_last:
            
            image = self.always_last(image)

        return image
    
