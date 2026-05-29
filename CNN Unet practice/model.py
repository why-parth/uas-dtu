import torch
import torch.nn as nn
import torch.nn.functional as F

import load

class ConvConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    
    def __init__(
            self, in_ch=3, out_ch=1, features=[64, 128, 256, 512]
        ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down Part #

        for feature in features:
            self.downs.append(ConvConv(in_ch, feature))
            in_ch = feature
        
        # Up Part #

        for feature in reversed(features):
            self.ups.append(
                    nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
                )
            self.ups.append(
                    ConvConv(feature*2, feature)
                )
            in_ch = feature

        self.bottleneck = ConvConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skip = []

        for down in self.downs:
            x = down(x)
            skip.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)

        skip = skip[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            conc = skip[i//2]

            if (x.shape[2:] != conc.shape[2:]):
                x = F.interpolate(
                        x,
                        size=conc.shape[2:],
                        mode="bilinear",
                        align_corners=False
                    )

            conc = torch.cat( (conc, x), dim=1)
            x = self.ups[i+1](conc)

        return self.final_conv(x)
    