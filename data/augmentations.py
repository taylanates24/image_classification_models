import imgaug.augmenters as iaa
import numpy as np
import random
import cv2

class Augmentations:
    
    def __init__(self, opt, img_size=None) -> None:

        geo_aug = []
        color_aug = []
        kernel_aug = []
        del_aug = []
        all_aug = []

        self.aug_prob = opt.get('aug_prob')
        scale = opt.get('scale')
        some_of = opt.get('some_of')
        if scale:
            if scale[-1]:
                geo_aug.append(iaa.Affine(scale=scale[:2]))
                all_aug.append(iaa.Affine(scale=scale[:2]))

        brightness = opt.get('brightness')

        if brightness:
            if brightness[-1]:
                color_aug.append(iaa.AddToBrightness(brightness[:2]))
                all_aug.append(iaa.AddToBrightness(brightness[:2]))

        saturation = opt.get('saturation')

        if saturation:
            if saturation[-1]:
                color_aug.append(iaa.AddToSaturation(saturation[:2]))
                all_aug.append(iaa.AddToSaturation(saturation[:2]))

        hue = opt.get('hue')

        if hue:
            if hue[-1]:
                color_aug.append(iaa.AddToHue(hue[:2]))
                all_aug.append(iaa.AddToHue(hue[:2]))

        add_grayscale = opt.get('add_grayscale')

        if add_grayscale:
            if add_grayscale[-1]:
                color_aug.append(iaa.Grayscale(alpha=add_grayscale[:2]))
                all_aug.append(iaa.Grayscale(alpha=add_grayscale[:2]))

        motion_blur = opt.get('motion_blur')

        if motion_blur:
            if motion_blur[-1]:
                kernel_aug.append(iaa.MotionBlur(k=motion_blur[:2]))
                all_aug.append(iaa.MotionBlur(k=motion_blur[:2]))

        translate = opt.get('translate')

        if translate:
            if translate[-1]:
                geo_aug.append(iaa.Affine(translate_percent={"x": translate[0], "y": translate[1]}))
                all_aug.append(iaa.Affine(translate_percent={"x": translate[0], "y": translate[1]}))

        rotate = opt.get('rotate')

        if rotate:
            if rotate[-1]:
                geo_aug.append(iaa.Affine(rotate=rotate[:2]))
                all_aug.append(iaa.Affine(rotate=rotate[:2]))

        shear = opt.get('shear')

        if shear:
            if shear[-1]:
                geo_aug.append(iaa.Affine(shear=shear[:2]))
                all_aug.append(iaa.Affine(shear=shear[:2]))

        contrast = opt.get('contrast')

        if contrast:
            if contrast[-1]:
                color_aug.append(iaa.LinearContrast(contrast))
                all_aug.append(iaa.LinearContrast(contrast))

        self.is_aug = len(geo_aug) or len(color_aug) or len(kernel_aug) or len(del_aug) or len(all_aug)

        self.fliplr = None
        self.crop = None
        self.resize = None
        self.center_crop = None

        self.geo_aug_seq  = None
        self.color_aug_seq = None
        self.kernel_aug_seq = None
        self.del_aug_seq  = None
        self.all_aug_seq = None

        if len(all_aug):
            self.all_aug_seq = iaa.SomeOf(n=some_of, children=all_aug, random_order=True)

        if len(geo_aug):
            self.geo_aug_seq = iaa.SomeOf(n=1, children=geo_aug, random_order=True)

        if len(color_aug):
            self.color_aug_seq = iaa.SomeOf(n=1, children=color_aug, random_order=True)

        if len(kernel_aug):
            self.kernel_aug_seq = iaa.SomeOf(n=1, children=kernel_aug, random_order=True)

        if len(del_aug):
            self.del_aug_seq = iaa.SomeOf(n=1, children=del_aug, random_order=True)


        fliplr = opt.get('fliplr')

        if fliplr:
            if fliplr[-1]:
                self.fliplr = iaa.Fliplr(fliplr[0])


        crop = opt.get('crop')

        if crop:
            if crop[-1]:
                self.crop = iaa.Crop(percent=tuple(crop[:2]))

        resize = opt.get('resize')

        if resize:
            if resize[-1]:
                #self.resize = iaa.Resize({"height": resize[0], "width": resize[1]}) if img_size is None else iaa.Resize({"height": img_size, "width": img_size})
                self.resize = cv2.resize
                self.height = resize[0] if img_size is None else img_size
                self.width = resize[1] if img_size is None else img_size

        center_crop = opt.get('center_crop')

        if center_crop:
            if center_crop[-1]:
                self.center_crop = self.random_center_crop
                self.size_prop = tuple(center_crop[:2])
            

    def __call__(self, image):

        if self.crop:

            image = self.crop.augment_image(image)

        elif self.center_crop:

            image = self.random_center_crop(image, self.size_prop)


        if self.resize:
            
            #image = self.resize.augment_image(image)
            image = self.resize(image, (self.width, self.height), interpolation = cv2.INTER_CUBIC)

        if not (random.uniform(0,1) > (1 - self.aug_prob)):

            if self.fliplr:

                image = self.fliplr.augment_image(image)

            return image


        # if self.geo_aug_seq:

        #     image = self.geo_aug_seq(image=image.astype(np.uint8))
    
        # if self.color_aug_seq:

        #     image = self.color_aug_seq(image=image.astype(np.uint8))

        # if self.kernel_aug_seq:

        #     image = self.geo_aug_seq(image=image.astype(np.uint8))

        # if self.del_aug_seq:

        #     image = self.geo_aug_seq(image=image.astype(np.uint8))

        # if self.fliplr:

        #     image = self.fliplr.augment_image(image)

        if self.all_aug_seq:
            image = self.all_aug_seq.augment_image(image)

        return image
    
    def random_center_crop(self, image, img_size_crop):
        """
        Apply a random center crop to a single image using OpenCV. The crop size is a random proportion
        of the smaller dimension of the original image.

        Parameters:
        image: The image to crop (as a numpy array).
        img_size_crop (tuple): The list of proportion of the original image's smaller dimension to be used for cropping.

        Returns:
        np.ndarray: The cropped image.
        """
        min_size_prop, max_size_prop = img_size_crop
        # Determine the image's original size
        original_height, original_width = image.shape[:2]

        # Determine the smaller dimension of the image
        smaller_dimension = min(original_height, original_width)

        # Calculate a random crop size as a proportion of the smaller dimension
        proportion = np.random.uniform(min_size_prop, max_size_prop)
        crop_size = int(smaller_dimension * proportion)

        # Calculate the cropping boundaries
        start_x = (original_width - crop_size) // 2
        start_y = (original_height - crop_size) // 2
        end_x = start_x + crop_size
        end_y = start_y + crop_size

        # Perform the crop
        cropped_image = image[start_y:end_y, start_x:end_x]

        return cropped_image


    def resize_cv(h, w):

        return cv2.resize