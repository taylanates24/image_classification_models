import numpy as np
import cv2
import torch 

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


def crop_fn_pil(image, ratio, corner="center"):
    """
    Crop the center or a specified corner of a PIL image based on the given ratio.

    Args:
        image (PIL.Image.Image): The input image to crop.
        ratio (float): The ratio of the cropped image size relative to the original size (0 < ratio <= 1).
        corner (str): The corner to crop from ("center", "upper-left", "upper-right", "lower-left", "lower-right").

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
    elif corner == "upper-right":
        # Set the cropping box coordinates for upper-right crop
        left = width - crop_width
        top = 0
    elif corner == "lower-left":
        # Set the cropping box coordinates for lower-left crop
        left = 0
        top = height - crop_height
    elif corner == "lower-right":
        # Set the cropping box coordinates for lower-right crop
        left = width - crop_width
        top = height - crop_height
    else:
        raise ValueError("Invalid corner option. Use 'center', 'upper-left', 'upper-right', 'lower-left', or 'lower-right'.")

    right = left + crop_width
    bottom = top + crop_height

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image



def preprocess_ir_eo(image, crop_fn, center_crop_ratio, corner_crop_ratio, corner_crop_type=None, img_size=180):
    image = crop_fn(image, center_crop_ratio, 'center')
    if corner_crop_type is None:
        corner_crop_type = 'upper-left'
    image = crop_fn(image, corner_crop_ratio, corner_crop_type)
    image = cv2.resize(image, (img_size, img_size))
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.from_numpy(image).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).cuda()
    return image_tensor


def crop_fn_cv2(image, ratio, corner="center"):
    if not (0 < ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1.")
    height, width = image.shape[:2]
    crop_width = int(width * ratio)
    crop_height = int(height * ratio)
    if corner == "center":
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
    elif corner == "upper-left":
        left = 0
        top = 0
    elif corner == "upper-right":
        left = width - crop_width
        top = 0
    elif corner == "lower-left":
        left = 0
        top = height - crop_height
    elif corner == "lower-right":
        left = width - crop_width
        top = height - crop_height
    else:
        raise ValueError("Invalid corner option. Use 'center', 'upper-left', 'upper-right', 'lower-left', or 'lower-right'.")
    right = left + crop_width
    bottom = top + crop_height
    return image[top:bottom, left:right]