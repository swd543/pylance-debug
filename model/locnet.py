import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LocNet(nn.Module):
    """Some Information about LocNet"""
    def __init__(self):
        super(LocNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(256, 128, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.Upsample(size=(252, 252)),
            nn.ConvTranspose2d(16, 1, 5, 2, output_padding=1, dilation=2),
            nn.ReLU()
        )

    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

if __name__ == "__main__":
    model = LocNet().cuda()
    print(model)
    r = torch.rand(12, 1, 512, 512).cuda()
    model(r)