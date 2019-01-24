import torch
import torch.nn as nn
from models.layers.unet_layer import UnetConv3D, UnetUpConv3D, weights_init_kaiming
from models.layers.GCN_layer import ResBlock, GCN, ConBR
#from layers.unet_layer import UnetConv3D, UnetUpConv3D, weights_init_kaiming
#from layers.GCN_layer import ResBlock, GCN, ConBR
import torch.nn.functional as F

#__init__(self, n_layers, in_c, mid_c, out_c, stride, padding)
class Unet3D_glob(nn.Module):

    def __init__(self, feature_scale=1, trans_feature=32, is_batchnorm=True):
        super(Unet3D_glob, self).__init__()
        layer_num = [2, 3, 3, 2]
        feature_num = [64, 128, 256, 512, 1024]
        feature_num = [x // feature_scale for x in feature_num]
        self.trans_feature=trans_feature

        # downsampling
        self.conv1 = UnetConv3D(1, feature_num[0], is_batchnorm) ## O : feature_num[0] * 64*64*32
        self.layer1 = ResBlock(layer_num[0], feature_num[0], feature_num[0]//2, feature_num[1], stride=2, padding=1) ## feature_num[1] * 32*32*16
        self.layer2 = ResBlock(layer_num[1], feature_num[1], feature_num[1]//2, feature_num[2], stride=2, padding=1) ## feature_num[2] * 16*16*8
        self.layer3 = ResBlock(layer_num[2], feature_num[2], feature_num[2]//2, feature_num[3], stride=2, padding=1) ## feature_num[3] * 8*8*4
        self.layer4 = ResBlock(layer_num[3], feature_num[3], feature_num[3]//2, feature_num[4], stride=2, padding=1) ## feature_num[4] * 4*4*2

        self.gcn1 = GCN(feature_num[0], self.trans_feature, k=(21, 21, 13))
        self.gcn2 = GCN(feature_num[1], self.trans_feature, k=(11, 11, 7)) #gcn_i after layer-1
        self.gcn3 = GCN(feature_num[2], self.trans_feature, k=(7, 7, 5))
        self.gcn4 = GCN(feature_num[3], self.trans_feature, k=(3, 3, 3))
        self.gcn5 = GCN(feature_num[4], self.trans_feature, k=(3, 3, 1))

        self.ConBR1=ConBR(2*self.trans_feature, self.trans_feature)
        self.ConBR2=ConBR(2*self.trans_feature, self.trans_feature)
        self.ConBR3=ConBR(2*self.trans_feature, self.trans_feature)
        self.ConBR4 =ConBR(2 * self.trans_feature, self.trans_feature)


        # final conv (without any concat)
        self.ConBRF=ConBR(self.trans_feature, self.trans_feature)

        self.final = UnetConv3D(self.trans_feature, self.trans_feature, is_batchnorm)
        self.final2 = nn.Conv3d(self.trans_feature,1,1)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)



    def forward(self, x):
        fm1 = self.conv1(x)
        fm2 = self.layer1(fm1)
        fm3 = self.layer2(fm2)
        fm4 = self.layer3(fm3)
        fm5 = self.layer4(fm4)

        gc_fm1 = self.gcn1(fm1)
        gc_fm2 = self.gcn2(fm2)
        gc_fm3 = self.gcn3(fm3)
        gc_fm4 = self.gcn4(fm4)
        gc_fm5 = self.gcn5(fm5)

        gc_fm5 = F.upsample(gc_fm5, fm4.size()[2:], mode='trilinear', align_corners=True)
        gc_fm4 = self.ConBR4(torch.cat((gc_fm4,gc_fm5),1))
        gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='trilinear', align_corners=True)
        gc_fm3 = self.ConBR3(torch.cat((gc_fm3,gc_fm4),1))
        gc_fm3 = F.upsample(gc_fm3, fm2.size()[2:], mode='trilinear', align_corners=True)
        gc_fm2 = self.ConBR2(torch.cat((gc_fm2, gc_fm3), 1))
        gc_fm2 = F.upsample(gc_fm2, fm1.size()[2:], mode='trilinear', align_corners=True)
        gc_fm1 = self.ConBR1(torch.cat((gc_fm1, gc_fm2), 1))
        gc_fm1 = F.upsample(gc_fm1, x.size()[2:], mode='trilinear', align_corners=True)
        ##추가
        final=self.final(gc_fm1)
        final = self.final2(final)

        return final

if __name__ == "__main__":
    input2D = torch.Tensor(1, 1, 128, 112, 80)
    model = Unet3D_glob()
    output2D = model(input2D)
    
    print("input shape : \t", input2D.shape)
    print("output shape  : \t", output2D.shape)




