import torch
import torch.nn as nn
from models.layers.unet_layer import UnetConv3D, UnetUpConv3D, weights_init_kaiming
import torch.nn.functional as F

class UnetR3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1,
                 is_deconv=True, is_batchnorm=True):
        super(UnetR3D, self).__init__()
        filters = [64, 128, 256, 512]
        filters = [x // feature_scale for x in filters]

        # downsampling
        self.conv1    = UnetConv3D(1, filters[0], is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2    = UnetConv3D(filters[0], filters[1], is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3    = UnetConv3D(filters[1], filters[2], is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.center   = UnetConv3D(filters[2], filters[3], is_batchnorm)

        # upsampling
        self.up_concat3 = UnetUpConv3D(filters[3], filters[2], is_deconv)
        self.up_concat2 = UnetUpConv3D(filters[2], filters[1], is_deconv)
        self.up_concat1 = UnetUpConv3D(filters[1], filters[0], is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.final2 = nn.Softmax(dim=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)


    def forward(self, inputs):
        conv1    = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2    = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3    = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        center = self.center(maxpool3)

        up3 = self.up_concat3(conv3, center)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final

if __name__ == "__main__":
    input2D = torch.Tensor(1, 1, 448, 448)
    model = Unet2D()
    output2D = model(input2D)
    
    print("input shape : \t", input2D.shape)
    print("output shape  : \t", output2D.shape)




