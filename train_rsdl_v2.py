import torch.nn as nn
import torch
import gdal
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

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


class RSDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images = self.read_multiband_images(images_dir)
        self.labels = self.read_singleband_labels(labels_dir)

    def read_multiband_images(self, images_dir):
        images = []
        for image_path in images_dir:
            rsdl_data = gdal.Open(image_path)
            images.append(np.stack([rsdl_data .GetRasterBand(i).ReadAsArray() for i in range(1, 4)], axis=0))
        return images

    def read_singleband_labels(self, labels_dir):
        labels = []
        for label_path in labels_dir:
            rsdl_data = gdal.Open(label_path)
            labels.append(rsdl_data .GetRasterBand(1).ReadAsArray())
        return labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label)

images_dir = ['data/2_95_sat.tif', 'data/2_96_sat.tif',  'data/2_97_sat.tif', 'data/2_98_sat.tif', 'data/2_976_sat.tif']
labels_dir =['data/2_95_mask.tif', 'data/2_96_mask.tif',  'data/2_97_mask.tif', 'data/2_98_mask.tif', 'data/2_976_mask.tif']

dataset = RSDataset(images_dir, labels_dir)
trainloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet(3, 1)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

num_epochs = 20

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        images = images.float()
        labels = labels.float() / 255.0
        outputs = model(images)
        labels = labels.squeeze(0)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    print('Epoch [{}/{}], Loss: {:.4f}, Learning Rate: {:.6f}'.format(epoch + 1, num_epochs, loss.item(), optimizer.param_groups[0]['lr']))

torch.save(model.state_dict(), 'models_building_20.pth')
