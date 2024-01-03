from .Xception import Xception
from torchvision.models import resnet50
from .efficientnet import EfficientNet
from torch import nn
import torch



num_classes = 2


def resnet_model(pretrained=None):
    model = resnet50()
    model.fc = nn.Linear(in_features=2048,out_features=num_classes,bias=True)
    if pretrained:
        model = model.load_state_dict(torch.load(pretrained))
    return model

def xception_model(pretrained=None):
    model = Xception(num_classes=num_classes)
    if pretrained:
        model = model.load_state_dict(torch.load(pretrained))
    return model

def efficientnet_model(pretrained=None):
    model = EfficientNet.from_name(model_name='efficientnet-b4',num_classes=num_classes)
    if pretrained:
        model = model.load_state_dict(torch.load(pretrained))
    return model


model_dic = {
    'xception':xception_model(),
    'resnet':resnet_model(),
    'efficientnet':efficientnet_model(),
}


if __name__ =='__main__':
    # model = resnet50(pretrained=False)
    # model.fc = nn.Linear(in_features=2048,out_features=2,bias=True) 
    # print(model.fc)
    # model = EfficientNet.from_name(model_name='efficientnet-b5',num_classes=2)
    # data = torch.rand([10,3,244,244])
    # device = torch.device('cuda:0')
    # data = data.to(device)
    # model.to(device)
    # a = model(data)
    pass