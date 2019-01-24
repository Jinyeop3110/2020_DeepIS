import torch
import torch.nn as nn
from models.layers.unet_layer import UnetConv3D, UnetUpConv3D, weights_init_kaiming
from models.layers.GCN_layer import ResBlock, GCN, ConBR
#from layers.unet_layer import UnetConv3D, UnetUpConv3D, weights_init_kaiming
#from layers.GCN_layer import ResBlock, GCN, ConBR
import torch.nn.functional as F

#__init__(self, n_layers, in_c, mid_c, out_c, stride, padding)
class Unet3D_glob2(nn.Module):

    def __init__(self, feature_scale=1, trans_feature=64, is_batchnorm=True):
        super(Unet3D_glob2, self).__init__()
        layer_num = [3, 3, 4, 3]
        feature_num = [64, 128, 256, 512, 1024]
        feature_num = [x // feature_scale for x in feature_num]
        self.trans_feature=trans_feature

        # downsampling
        self.conv1 = UnetConv3D(1, feature_num[0], is_batchnorm) ## O : feature_num[0] * 64*64*32
        self.layer1 = ResBlock(layer_num[0], feature_num[0], feature_num[0]//2, feature_num[1], stride=2, padding=1) ## feature_num[1] * 32*32*16
        self.layer2 = ResBlock(layer_num[1], feature_num[1], feature_num[1]//2, feature_num[2], stride=2, padding=1) ## feature_num[2] * 16*16*8
        self.layer3 = ResBlock(layer_num[2], feature_num[2], feature_num[2]//2, feature_num[3], stride=2, padding=1) ## feature_num[3] * 8*8*4
        self.layer4 = ResBlock(layer_num[3], feature_num[3], feature_num[3]//2, feature_num[4], stride=2, padding=1) ## feature_num[4] * 4*4*2

        self.gcn1 = GCN(feature_num[1], self.trans_feature, k=(7, 7, 5)) #gcn_i after layer-1
        self.gcn2 = GCN(feature_num[2], self.trans_feature, k=(7, 7, 5))
        self.gcn3 = GCN(feature_num[3], self.trans_feature, k=(5, 5, 3))
        self.gcn4 = GCN(feature_num[4], self.trans_feature, k=(3, 3, 1))

        self.up1 = nn.ConvTranspose3d(2*self.trans_feature, self.trans_feature, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.up2 = nn.ConvTranspose3d(2*self.trans_feature, self.trans_feature, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.up3 = nn.ConvTranspose3d(2*self.trans_feature, self.trans_feature, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

        # final conv (without any concat)
        #self.final = nn.Conv3d(self.trans_feature,1, 1)
        self.BR=ConBR(self.trans_feature, self.trans_feature)
        self.final = nn.Conv3d(self.trans_feature,1, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)



    def forward(self, x):
        x = self.conv1(x)
        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm1 = self.gcn1(fm1)
        gc_fm2 = self.gcn2(fm2)
        gc_fm3 = self.gcn3(fm3)
        gc_fm4 = self.gcn4(fm4)

        gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='trilinear', align_corners=True)


        gc_fm3 = torch.cat((gc_fm3,gc_fm4),1)
        gc_fm3 = self.up3(gc_fm3)
        gc_fm2 = torch.cat((gc_fm2, gc_fm3), 1)
        gc_fm2 = self.up2(gc_fm2)
        gc_fm1 = torch.cat((gc_fm1, gc_fm2), 1)
        gc_fm1 = self.up1(gc_fm1)
        final = self.BR(gc_fm1)
        final = self.final(final)

        return final

if __name__ == "__main__":
    input2D = torch.Tensor(1, 1, 128, 112, 80)
    model = Unet3D_glob()
    output2D = model(input2D)
    
    print("input shape : \t", input2D.shape)
    print("output shape  : \t", output2D.shape)




