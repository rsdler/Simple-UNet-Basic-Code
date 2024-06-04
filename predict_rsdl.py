import torch.nn as nn
import gdal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class UNet(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.center = self.conv_block(512, 1024)
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64,out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        center = self.center(self.pool(enc4))

        dec4 = self.dec4(torch.cat([enc4, self.up(center)], 1))
        dec3 = self.dec3(torch.cat([enc3, self.up(dec4)], 1))
        dec2 = self.dec2(torch.cat([enc2, self.up(dec3)], 1))
        dec1 = self.dec1(torch.cat([enc1, self.up(dec2)], 1))
        final = self.final(dec1).squeeze()

        return torch.sigmoid(final)

model = UNet(3, 1)
model.load_state_dict(torch.load('models_building_50.pth'))
model.eval()

image_file='data/2_955_sat.tif'
rsdataset = gdal.Open(image_file)
images=(np.stack([rsdataset.GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0))
test_images = torch.tensor(images).float().unsqueeze(0)

outputs = model(test_images)
outputs = (outputs > 0.8).float()

cv2.imshow('Prediction', outputs.numpy())
cv2.waitKey(0)
