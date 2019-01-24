import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndimage
import os
import scipy.misc
from glob import glob
from scipy import io


class loss_function_VAE(nn.Module):
    def __init__(self, N=16):
        super().__init__()
        self.N=N

    def forward(self, recon_x, x, mu, logvar):
        BCE = torch.sum(torch.pow(recon_x - x.view(-1, self.N * self.N), 2))
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

if __name__ == "__main__":
    img_paths=["/data1/0000Deployed_JY/strain_/20180416_20180416.131920.821.CD4-K562-003_000000_segmented_auto_001_.mat",
                "/data1/0000Deployed_JY/strain_/20180416_20180416.134340.619.CD4-K562-008_000000_segmented_auto_001_.mat",
                "/data1/0000Deployed_JY/strain_/test002_tomogram_000009_002_.mat",
               "/data1/0000Deployed_JY/strain_/negative011_tomogram_000012_002_.mat",
                "/data1/0000Deployed_JY/strain_/test002_tomogram_000026_001_.mat",
               "/data1/0000Deployed_JY/strain_/20180521_20180521.193521.312.stimCD8-KwithP-046_000000_001_.mat",
               "/data1/0000Deployed_JY/strain_/test007_tomogram_000018_002_.mat"
               ]
    i=0
    for img_path in img_paths:
        i=i+1
        data_pack = io.loadmat(img_path)
        # 2D ( 1 x H x W )
        input_np = data_pack['input']
        target_np = data_pack['target']

        input = torch.from_numpy(input_np).view(1, 1, 128, 128, 64).to(torch.float)
        target=torch.from_numpy(target_np).view(1,1,128,128,64).to(torch.float)
        threshold = 0.05
        erode=4
        version=2


        if (version == 2):
            weight_mask1 = (target > threshold).to(torch.float)
            kernel = torch.ones(1, 1, 5, 5, 5).to()
            weight_mask1 = F.conv3d(weight_mask1, kernel, padding=2)
            weight_mask1 = ((weight_mask1) > 0).to(torch.float)

            weight_mask2 = (target < -threshold).to(torch.float)
            kernel = torch.ones(1, 1, 5, 5, 5).to()
            weight_mask2 = F.conv3d(weight_mask2, kernel, padding=2)
            weight_mask2 = ((weight_mask2) > 0).to(torch.float)

            weight_mask = weight_mask1 * weight_mask2
            kernel = torch.ones(1, 1, 2 * erode + 1, 2 * erode + 1, 2 * erode + 1).to()
            weight_mask = F.conv3d(weight_mask, kernel, padding=erode).to(torch.float)
            weight_mask = 1 + weight_mask * 50. / pow(2 * erode + 1, 3)
        else:
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        weight_mask = weight_mask + 5 * weight_mask1 + 15 * weight_mask2
        loss = torch.pow(input - target,2)
        loss = (loss * weight_mask).mean()



        data = {}
        data['input'] = input_np
        data['loss'] = loss.numpy()
        data['target']=target_np
        data['weight_mask'] = torch.squeeze(weight_mask).numpy()
        scipy.io.savemat("/home/jysong/PyCharmProjects_JY/181007_3DcellSegmentation_regressionVer_TBdistinguish/losstest" + "/losstest%04d.mat" %(i), data)
