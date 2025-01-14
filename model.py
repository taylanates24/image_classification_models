import torch.nn as nn
import timm


class SceneClassifier(nn.Module):
    
    def __init__(self, model_name='tf_efficientnet_lite0', pretrained=True, num_classes=3):
        # eva02_tiny_patch14_336.mim_in22k_ft_in1k with imagesize 336
        # 'fastvit_s12' 8.5 M
        # 'fastvit_sa12' 10.6 M
        # 'fastvit_sa24' 20.5 M
        # 'fastvit_sa36' 30.5 M
        #  'fastvit_t8' 3.3 M
        # 'fastvit_t12' 6.5 M
        #  fastvit_ma36 42.9 M

        # 'efficientvit_b0' 2.1 M
        # 'efficientvit_b1' 7.5 M 
        # 'efficientvit_b2' 21.8 M
        # 'efficientvit_b3' 46.1 M
        # 'efficientvit_l1' 49.5 M
        # 'efficientvit_l2'
        # 'efficientvit_l3'

        # 'efficientformer_l1'
        # 'efficientformer_l3'
        # 'efficientformer_l7'
        # 'efficientformerv2_l'
        # 'efficientformerv2_s0' 140 img size
        # 'efficientformerv2_s1'
        # 'efficientformerv2_s2'

        # mixnet_s 2.6 M

        # 'mobilevit_s' 4.9 M
        # 'mobilevit_xs' 1.9 M
        # 'mobilevit_xxs' 951 K
        # 'mobilevitv2_050' 1.1 M
        # 'mobilevitv2_075' 2.5 M
        # 'mobilevitv2_100'


        # 'efficientnet_cc_b0_4e'
        # 'efficientnet_cc_b0_8e'
        # 'efficientnet_cc_b1_8e'

        # 'flexivit_small'
        super(SceneClassifier, self).__init__()
        avail_pretrained_models = timm.list_models(pretrained=True)
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, image):
        
        return self.model(image)
    