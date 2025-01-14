from model import SceneClassifier
import torch
from torch2trt import torch2trt
import argparse
import os

def convert_tensorrt(ckpt_pth, num_classes=1, input_size=128, save=True, save_path=None, model_name='mobilevitv2_200'):

    if save_path is None:

        save_path = ckpt_pth.replace('weights', 'tensorrt_weights').rsplit('.', 1)[0] + '.trt'

        dir_path = os.path.dirname(save_path)

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    from torch2trt import tensorrt_converter
    import tensorrt as trt

    import torch.nn.functional as F

    @tensorrt_converter('torch.nn.functional.hardtanh')
    def convert_hardtanh(ctx):
        input = ctx.method_args[0]
        min_val = ctx.method_kwargs.get('min_val', -1.0)
        max_val = ctx.method_kwargs.get('max_val', 1.0)
        
        output = ctx.method_return
        
        # Create TensorRT layer for clamping (equivalent to HardTanh)
        layer = ctx.network.add_activation(input._trt, trt.ActivationType.CLIP)
        layer.alpha = min_val
        layer.beta = max_val
        
        output._trt = layer.get_output(0)
    


    checkpoint = torch.load(ckpt_pth)
    model_state_dict = checkpoint['model_state_dict']
    model = SceneClassifier(pretrained=False, num_classes=num_classes, model_name=model_name)

    model.load_state_dict(model_state_dict)

    print('The pretrained model is loaded.')

    image = torch.randn(1, 3, input_size, input_size).cuda()
    model.eval().cuda()

    model_trt = torch2trt(model, [image], fp16_mode=True)
    print('The model is converted to TensorRT.')

    import time

    torch_model_time = []

    for i in range(100):
        st = time.time()
        model(image)
        end = time.time()
        torch_model_time.append(end-st)

    trt_model_time = []

    for i in range(100):
        st = time.time()
        model_trt(image)
        end = time.time()
        trt_model_time.append(end-st)

    print('trt output: ', model_trt(image), 'time: ', 1000 * (sum(trt_model_time) / len(trt_model_time)))
    print('model output: ', model(image), 'time: ', 1000 * (sum(torch_model_time) / len(torch_model_time)))
    

    if save:
        torch.save(model_trt.state_dict(), save_path)
        print('The model is converted to TensorRT.')

    return model_trt, save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default='experiments/model_mobilevitv2_200_bs_2_img_size_224_opt_adam_loss_CE_date_2025-01-10 13:58:54.638655/weights/best_epoch_2_precision_99.ckpt', 
                        help='checkpoint path of torch model')   
    parser.add_argument('--ex_img_path', type=str, 
                        default='video_type_dataset/RGB/1_0_305.jpg', 
                        help='checkpoint path of torch model')
    parser.add_argument('--num_classes', type=int, 
                        default=2, 
                        help='number of classes') 
    parser.add_argument('--input_size', type=int, 
                        default=224, 
                        help='the size of the input image') 
    parser.add_argument('--save_path', type=str, 
                        default='deneme_ir.pth', 
                        help='the path of tensorrt model')  
    
    args = parser.parse_args()
    ex_img_path = args.ex_img_path
    convert_tensorrt(ckpt_pth=args.checkpoint,
                     num_classes=args.num_classes,
                     input_size=args.input_size,
                     save_path=args.save_path)

