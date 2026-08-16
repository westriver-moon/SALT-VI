import torch
import torchvision
import torch.nn as nn

class RGB_Model(nn.Module):
    def __init__(self, pretrain_path="default"):
        super(RGB_Model, self).__init__()
        if pretrain_path == "default":
            resnet = torchvision.models.resnet50(pretrained=True)
        else:
            resnet = torchvision.models.resnet50(pretrained=False)
            resnet.load_state_dict(torch.load(pretrain_path),strict=False)


        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)

    def forward(self, rgb):
        rgb_features_map = self.resnet_conv(rgb)
        return rgb_features_map

class IR_Model(nn.Module):
    def __init__(self, pretrain_path=None):
        super(IR_Model, self,).__init__()
        if pretrain_path == "default":
            resnet = torchvision.models.resnet50(pretrained=True)
        else:
            resnet = torchvision.models.resnet50(pretrained = False)
            resnet.load_state_dict(torch.load(pretrain_path),strict=False)

        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)

    def forward(self, ir):
        ir_features_map = self.resnet_conv(ir)
        return ir_features_map

class Shared_Model(nn.Module):

    def __init__(self, pretrain_path=None):
        super(Shared_Model, self,).__init__()
        if pretrain_path == "default":
            resnet = torchvision.models.resnet50(pretrained=True)
        else:
            resnet = torchvision.models.resnet50(pretrained = False)
            resnet.load_state_dict(torch.load(pretrain_path),strict=False)
        
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.resnet_conv = nn.Sequential(resnet.layer1,
                                         resnet.layer2, resnet.layer3, resnet.layer4)

    def forward(self, x):
        features_map = self.resnet_conv(x)
        return features_map


