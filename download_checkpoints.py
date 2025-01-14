import torch
import timm
import os
from tqdm import tqdm

model_list = ['cspresnet50.ra_in1k', 'resnet18.a1_in1k', 'cspdarknet53.ra_in1k', 'fastvit_s12.apple_dist_in1k', 'fastvit_t8.apple_dist_in1k',
              'fastvit_t12.apple_dist_in1k', 'hgnet_tiny.paddle_in1k', 'hgnetv2_b0.ssld_stage1_in22k_in1k',
              'hgnetv2_b1.ssld_stage1_in22k_in1k', 'hgnetv2_b2.ssld_stage1_in22k_in1k', 'hgnetv2_b3.ssld_stage1_in22k_in1k',
              'inception_v4.tf_in1k', 'mobileone_s0.apple_in1k', 'resnet10t.c3_in1k', 'resnest14d.gluon_in1k',
              'resnest26d.gluon_in1k', 'resnet14t.c3_in1k', 'resnet32ts.ra2_in1k',
              'resnet34.a1_in1k', 'resnext26ts.ra2_in1k', 'resnext50_32x4d.a1_in1k', 'tinynet_e.in1k', 'tinynet_d.in1k',
              'tinynet_b.in1k', ]
model_path = 'pretrained_models'
if not os.path.isdir(model_path):
    os.makedirs(model_path)
for model_name in tqdm(model_list):
    # model = timm.create_model(model_name, pretrained=True, num_classes=2)
    #torch.save({'model_state_dict': model.state_dict()}, f'{model_path}/{model_name}.ckpt')
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=2)
        torch.save({'model_state_dict': model.state_dict(),}, f'{model_path}/{model_name}.ckpt')
    except:
        print('error')
    
print('asd')