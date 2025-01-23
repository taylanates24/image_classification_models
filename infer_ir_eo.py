import argparse
import cv2
import json
import os
import time
from tqdm import tqdm
from infer import Infer
from data.utils import preprocess_ir_eo, crop_fn_cv2

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='resnet10t.pth', 
                        help='checkpoint path of tensorrt model')
    parser.add_argument('--images_path', type=str, default='/dataset/images_all', 
                        help='the path of tensorrt model')
    parser.add_argument('--ann_path', type=str, default='/dataset/annotations/all.json', 
                        help='the path of tensorrt model')
    parser.add_argument('--out_ann_path', type=str, default='annotations_updated.json', 
                        help='the path of tensorrt model')
    parser.add_argument('--input_size', type=int, default=180, 
                        help='the size of the input image')
    parser.add_argument('--score_thr', type=float, default=0.5, 
                        help='score threshold for classification')
    return parser.parse_args()

def main():
    args = arg_parser()

    images_path = args.images_path
    ann_path = args.ann_path
    infer = Infer(args.checkpoint)
    score_thr = args.score_thr
    input_size = args.input_size

    # Load and process annotation data
    with open(ann_path, 'r') as f:
        data = json.load(f)

    images_all = data['images']
    times = []

    # Process each image
    for i, image_ann in enumerate(tqdm(images_all)):
        file_name = image_ann['file_name']
        image_path = os.path.join(images_path, file_name)
        image = cv2.imread(image_path)

        # Preprocess image
        st = time.time()
        image = preprocess_ir_eo(
            image=image, 
            crop_fn=crop_fn_cv2, 
            center_crop_ratio=0.6, 
            corner_crop_ratio=0.5, 
            img_size=input_size,
            corner_crop_type='upper-left'
        )

        # Perform inference
        result = infer.inference(image)
        end = time.time()
        times.append(end - st)

        # Update image type based on result
        image_ann['type'] = 0 if result > score_thr else 1

    # Update data and calculate average processing time
    data['images'] = images_all
    avg_time = (sum(times) / len(times)) * 1000
    print(f'Average time per image: {avg_time:.2f} ms')

    # Save updated annotations
    with open(args.out_ann_path, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()
