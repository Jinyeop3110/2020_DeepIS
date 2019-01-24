from multiprocessing import Process
import os
import argparse
import torch
import torch.nn as nn
import sys
import utils
torch.backends.cudnn.benchmark = True

# example for mnist
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, '/home/Jinyeop/PyCharmProjects_JY/180801_2DcellSegmentation_JY')

#from slack_server import SlackBot
import datas.preprocess as preprocess

from Logger import Logger

from models.Fusionnet import Fusionnet
from models.unet import Unet3D
from models.unet_reduced import UnetR3D
from models.unet_glob1007 import Unet3D_glob
from models.unet_glob1106 import Unet3D_glob1106
from models.unet_glob1109 import Unet3D_glob1109
from models.unet_glob1111 import Unet3D_glob1111

from models.unet_glob2 import Unet3D_glob2
from datas.NucleusLoader2 import nucleusloader
from datas.NucleusLoader3 import nucleusloader3
from trainers.CNNTrainer2 import CNNTrainer
from trainers.CNNTrainer3 import CNNTrainer1106
import copyreg
from scipy import io


from loss import FocalLoss, TverskyLoss, FocalLoss3d_ver1, FocalLoss3d_ver2, DiceDis, DicePlusDis, Classifying, Classifying2

"""parsing and configuration"""
def arg_parse():
    # projects description
    desc = "TBcell 2D segmentation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="8",
                        help="Select CPU Number workers")
    # Unet params
    parser.add_argument('--feature_scale', type=int, default=4)

    parser.add_argument('--in_channel', type=int, default=1)

    # FusionNet Parameters
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--clamp', type=tuple, default=None)

    parser.add_argument('--augment', type=str, default='',
                        help='The type of augmentaed ex) crop,rotate ..  | crop | flip | elastic | rotate |')

    # TODO : Weighted BCE
    parser.add_argument('--loss', type=str, default='l1',
                        choices=["l1", "l2"])

    parser.add_argument('--epoch', type=int, default=500, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='The size of batch')
    parser.add_argument('--test', type=int, default=0, help='The size of batch')

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')

    # Adam Parameter
    parser.add_argument('--lrG', type=float, default=0.0001)
    parser.add_argument('--beta', nargs="*", type=float, default=(0.5, 0.999))

    return parser.parse_args()


def reconstruct_torch_dtype(torch_dtype: str):
    # a dtype string is "torch.some_dtype"
    dtype = torch_dtype.split('.')[1]
    return getattr(torch, dtype)


def pickle_torch_dtype(torch_dtype: torch.dtype):
    return reconstruct_torch_dtype, (str(torch_dtype),)

if __name__ == "__main__":
    arg = arg_parse()
    arg.save_dir = "1120_basic"


    arg.save_dir = "%s/outs/%s"%(os.getcwd(), arg.save_dir)
    if os.path.exists(arg.save_dir) is False:
            os.mkdir(arg.save_dir)
    
    logger = Logger(arg.save_dir)

    copyreg.pickle(torch.dtype, pickle_torch_dtype)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    preprocess = preprocess.get_preprocess(arg.augment)

    train_loader = nucleusloader(f_path_train, arg.batch_size, transform=preprocess,
                                 cpus=arg.cpus,
                                 shuffle=True, drop_last=True)

    valid_loader = nucleusloader3(f_path_valid, batch_size=1, transform=None,
                                 cpus=arg.cpus, shuffle=False,
                                drop_last=True)


    if arg.model == "fusion":
        net = Fusionnet(arg.in_channel, arg.out_channel, arg.ngf, arg.clamp)
    elif arg.model == "unet":
        net = Unet3D(feature_scale=arg.feature_scale)
    elif arg.model == "unet_gh":
        ## "nets_1004_unet_glob_absloss_FRE_pw10_erode2_feat1_trans30"
        #net = Unet3D_glob2(feature_scale=arg.feature_scale, trans_feature=64)
        #net = Unet3D_glob(feature_scale=arg.feature_scale, trans_feature=64)
        #net = Unet3D_glob1106(feature_scale=arg.feature_scale, trans_feature=64)
        #net = Unet3D_glob1106(feature_scale=arg.feature_scale, trans_feature=24)

        #for arg.save_dir = "nets_1110_unet_glob1109_class_FRE_pw1_erode2_feat2_trans24_signdiff_newdata"
        #net=Unet3D_glob1109(feature_scale=arg.feature_scale, trans_feature=24)
        #net = Unet3D_glob1109(feature_scale=arg.feature_scale, trans_feature=32)
        net=Unet3D_glob1111(feature_scale=arg.feature_scale, trans_feature=24)

    elif arg.model == "unet_reduced":
        net = UnetR3D(feature_scale=arg.feature_scale)
    else:
        raise NotImplementedError("Not Implemented Model")

    net = nn.DataParallel(net).to(torch_device)


    #past losses
    #recon_loss=DiceDis(ratio1=0, gamma=2, pw=10, erode=4, is_weight=1)
    #val_loss=DiceDis(ratio1=0, gamma=2, pw=10, erode=4, is_weight=1)

    #current losses
    #recon_loss=DicePlusDis(0.5)
    #val_loss = DicePlusDis(0.5)

    #current losses
    #recon_loss=Classifying(pw=50, erode=2, is_weight=1, version=2)
    #val_loss = Classifying(pw=50, erode=2, is_weight=1, version=2)

    recon_loss=Classifying2(pw=50, erode=2, is_weight=1, version=2)
    val_loss = Classifying2(pw=50, erode=2, is_weight=1, version=2)

    if(True):
        model = CNNTrainer1106(arg, net, torch_device, recon_loss=recon_loss, val_loss=val_loss, logger=logger)
        #model.load(filename="epoch[0402]_losssum[0.016887].pth.tar")
        filename="epoch[0647]_losssum[0.056025].pth.tar"
        model.load(filename=filename)
        model.best_metric=0.05806
        if arg.test==0:
            model.train(train_loader, valid_loader)
        if arg.test==1:
            model.test(valid_loader,savedir=filename+'validation')
            model.test(test_loader, savedir=filename +'hardcase')
        # utils.slack_alarm("zsef123", "Model %s Done"%(arg.save_dir))

    if(False):
        arg.save_dir = "epoch[0014]_losssum[0.212180].pth.tar"
        arg.save_dir = "%s/outs/%s" % (os.getcwd(), arg.save_dir)
        model = CNNTrainer(arg, net, torch_device, recon_loss=recon_loss, val_loss=val_loss, logger=logger)
        filename = "epoch[0324]_losssum[0.053456].pth.tar"
        model.load(filename=filename)
        test_loader = nucleusloader(f_path_valid, batch_size=1, transform=None,
                                    cpus=arg.cpus, shuffle=True,
                                    drop_last=True)
        model.test(test_loader,savedir='validset'+filename)

    if (False):
        arg.save_dir = "nets_1021_unet_glob1007_disdice_FRE_pw1_erode2_feat1_trans64_bothplus"
        arg.save_dir = "%s/outs/%s" % (os.getcwd(), arg.save_dir)
        model = CNNTrainer(arg, net, torch_device, recon_loss=recon_loss, val_loss=val_loss, logger=logger)
        filename = "epoch[0418]_losssum[0.049294].pth.tar(pw1.5)"
        model.load(filename=filename)
        test_loader = nucleusloader(f_path_valid, batch_size=1, transform=None,
                                    cpus=arg.cpus, shuffle=True,
                                    drop_last=True)
        model.test(test_loader, savedir='validset' + filename)

    if (False) :
        arg.save_dir = "nets_1013_unet_glob1007_disdice_FRE_pw1_erode2_feat1_trans64"
        arg.save_dir = "%s/outs/%s" % (os.getcwd(), arg.save_dir)
        model = CNNTrainer(arg, net, torch_device, recon_loss=recon_loss, val_loss=val_loss, logger=logger)
        filename = "epoch[0324]_losssum[0.053456].pth.tar"
        model.load(filename=filename)

        img_path = "/data1/0000Deployed_JY/strain_/test007_tomogram_000018_002_.mat"

        model.activationSave(img_path, savedir='activationtest' + filename)

