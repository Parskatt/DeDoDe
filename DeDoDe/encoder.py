import torch
import torch.nn as nn
import torchvision.models as tvm


class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast("cuda", enabled=self.amp, dtype = self.amp_dtype):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes

class VGG(nn.Module):
    def __init__(self, size = "19", pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if size == "11":
            self.layers = nn.ModuleList(tvm.vgg11_bn(pretrained=pretrained).features[:22])
        elif size == "13": 
            self.layers = nn.ModuleList(tvm.vgg13_bn(pretrained=pretrained).features[:28])
        elif size == "19": 
            self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast("cuda", enabled=self.amp, dtype = self.amp_dtype):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes
