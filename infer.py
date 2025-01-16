import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch2trt import TRTModule
from data.utils import crop_fn_cv2, preprocess_ir_eo

class Infer:
    def __init__(self, checkpoint_pth):
        """
        Initialize the Infer class with the TensorRT model and Softmax function.

        Args:
            checkpoint_pth (str): Path to the model checkpoint.
        """
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(checkpoint_pth))
        self.softmax = nn.Softmax(dim=1)

    def inference(self, image):
        """
        Perform inference on the input image.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            float: Softmax score of the first output class.
        """
        out = self.model(image)
        out_soft = self.softmax(out)
        return out_soft[0][0].item()

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='resnet10t.pth',
                        help='Checkpoint path of TensorRT model')
    parser.add_argument('--img_path', type=str, default='video_type_dataset/IR/frame1.jpg',
                        help='Path to the input image')
    parser.add_argument('--input_size', type=int, default=180,
                        help='Size of the input image')
    parser.add_argument('--score_thr', type=float, default=0.5,
                        help='Score threshold for classification')
    return parser.parse_args()

def main():
    """
    Main function to perform inference using the Infer class.
    """
    args = parse_arguments()

    # Initialize inference model
    infer = Infer(args.checkpoint)
    score_thr = args.score_thr

    # Load and preprocess the input image
    image = cv2.imread(args.img_path)
    image = preprocess_ir_eo(image=image, crop_fn=crop_fn_cv2, 
                             center_crop_ratio=0.6, corner_crop_ratio=0.5)

    # Perform inference
    result = infer.inference(image)

    # Convert result to binary classification
    result = 0 if result > score_thr else 1

    print(result)

if __name__ == '__main__':
    main()
