import argparse
import os
import cv2
import torch
from torch2trt import torch2trt, tensorrt_converter
from data.utils import preprocess_ir_eo, crop_fn_cv2
import tensorrt as trt
from model import SceneClassifier
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@tensorrt_converter('torch.nn.functional.hardtanh')
def convert_hardtanh(ctx):
    input = ctx.method_args[0]
    min_val = ctx.method_kwargs.get('min_val', -1.0)
    max_val = ctx.method_kwargs.get('max_val', 1.0)
    output = ctx.method_return
    layer = ctx.network.add_activation(input._trt, trt.ActivationType.CLIP)
    layer.alpha = min_val
    layer.beta = max_val
    output._trt = layer.get_output(0)

def compare_outputs(torch_output, trt_output):
    difference = torch.abs(torch_output - trt_output).max().item()
    ratio = torch.abs(torch_output - trt_output) / torch.clamp(torch.abs(torch_output), min=1e-6)
    max_ratio = ratio.max().item()

    logger.info(f"The difference of the outputs is: {difference:.6f}")
    logger.info(f"The maximum ratio of the differences to the outputs is: {max_ratio:.6f}")

def convert_tensorrt(ckpt_pth, image_path, num_classes=1, input_size=128, save=True, save_path=None, model_name='mobilevitv2_200', fp16=True):
    if save_path is None:
        save_path = ckpt_pth.replace('weights', 'tensorrt_weights').rsplit('.', 1)[0] + '.trt'
        dir_path = os.path.dirname(save_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    checkpoint = torch.load(ckpt_pth)
    model_state_dict = checkpoint['model_state_dict']
    model = SceneClassifier(pretrained=False, num_classes=num_classes, model_name=model_name)
    model.load_state_dict(model_state_dict)
    logger.info('The pretrained model is loaded.')

    image = cv2.imread(image_path)
    image = preprocess_ir_eo(image=image, 
                             crop_fn=crop_fn_cv2, 
                             center_crop_ratio=0.6, 
                             corner_crop_ratio=0.5, 
                             img_size=input_size)
    model.eval().cuda()

    model_trt = torch2trt(model, [image], fp16_mode=fp16)
    logger.info('The model is converted to TensorRT.')

    torch_model_time = []
    trt_model_time = []

    for _ in range(100):
        st = time.time()
        torch_output = model(image)
        torch_model_time.append(time.time() - st)

        st = time.time()
        trt_output = model_trt(image)
        trt_model_time.append(time.time() - st)

    logger.info(f'TRT output: {trt_output}, time: {1000 * (sum(trt_model_time) / len(trt_model_time)):.2f} ms')
    logger.info(f'Model output: {torch_output}, time: {1000 * (sum(torch_model_time) / len(torch_model_time)):.2f} ms')

    compare_outputs(torch_output, trt_output)

    speedup = (sum(torch_model_time) / len(torch_model_time)) / (sum(trt_model_time) / len(trt_model_time))
    logger.info(f'TensorRT model is {speedup:.2f} times faster than the Torch model.')

    if save:
        torch.save(model_trt.state_dict(), save_path)
        logger.info(f'The model is saved at {save_path}.')

    return model_trt, save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet10t.c3_in1k', help='The name of the torch model')
    parser.add_argument('--checkpoint', type=str, default="experiments/model_resnet10t.c3_in1k_bs_1_img_size_180_opt_adam_loss_CE_date_2025-01-16 10:40:32.209681/weights/best_epoch_19_precision_99.ckpt", help='Checkpoint path of the torch model')
    parser.add_argument('--ex_img_path', type=str, default='video_type_dataset/IR/frame7.jpg', help='Example image path for conversion')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--fp16', type=bool, default=True, help='FP16 mode of tensorrt')
    parser.add_argument('--input_size', type=int, default=180, help='The size of the input image')
    parser.add_argument('--save_path', type=str, default='resnet10t2.pth', help='Path to save the TensorRT model')

    args = parser.parse_args()
    convert_tensorrt(
        ckpt_pth=args.checkpoint,
        num_classes=args.num_classes,
        input_size=args.input_size,
        save_path=args.save_path,
        model_name=args.model_name,
        image_path=args.ex_img_path,
        fp16=args.fp16
    )
