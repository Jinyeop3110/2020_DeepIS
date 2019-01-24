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
from models.unet_glob2 import Unet3D_glob2
from models.unet_glob1106 import Unet3D_glob1106

from datas.NucleusLoader import nucleusloader
from datas.HelaLoader import Helaloader
from trainers.CNNTrainer2 import CNNTrainer
from trainers.BigTrainer import BigTrainer
import copyreg
from loss import FocalLoss, TverskyLoss, FocalLoss3d_ver1, FocalLoss3d_ver2, DiceDis

"""parsing and configuration"""
def arg_parse():
    # projects description
    desc = "TBcell 2D segmentation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="8",
                        help="Select CPU Number workers")
    parser.add_argument('--model', type=str, default='unet',
                        choices=['fusion', "unet", "unet_gh", "unet_reduced"], required=True)
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

    #parser.add_argument('--data', type=str, default='data',
    #                    choices=['All', 'Balance', 'data', "Only_Label"],
    #                    help='The dataset | All | Balance | Only_Label |')

    parser.add_argument('--epoch', type=int, default=500, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='The size of batch')
    parser.add_argument('--test', type=int, default=0, help='The size of batch')

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')

    # Adam Parameter
    parser.add_argument('--lrG', type=float, default=0.0005)
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
    #arg.save_dir = "nets_1013_unet_glob1007_disdice_FRE_pw1_erode2_feat1_trans64"
    arg.save_dir = "nets_1107_unet_glob1106_onlydis_FRE_pw1_erode2_feat4_trans32_signdiff_newdata"
    arg.save_dir = "%s/outs/%s"%(os.getcwd(), arg.save_dir)
    if os.path.exists(arg.save_dir) is False:
            os.mkdir(arg.save_dir)
    
    logger = Logger(arg.save_dir)

    copyreg.pickle(torch.dtype, pickle_torch_dtype)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")
    #torch_device = torch.device("cpu")

    # manually change paths of training data and test data
    # filename example :
    f_path_train="/home/jysong/PyCharmProjects_JY/180907_3DcellSegmentation_regressionVer/data/train_"
    #f_path_test = "/home/jysong/PyCharmProjects_JY/180907_3DcellSegmentation_regressionVer/data/test_"
    f_path_test = "/data1/0000Deployed_JY/hardcase"
    f_path_valid = "/data1/0000Deployed_JY/svalid_"
    #f_path_test = "/data1/Moosung_CART/For_analysis/dataset1/exp0_fullsequence"

    preprocess = preprocess.get_preprocess(arg.augment)


    if arg.model == "fusion":
        net = Fusionnet(arg.in_channel, arg.out_channel, arg.ngf, arg.clamp)
    elif arg.model == "unet":
        net = Unet3D(feature_scale=arg.feature_scale)
    elif arg.model == "unet_gh":
        ## "nets_1004_unet_glob_absloss_FRE_pw10_erode2_feat1_trans30"
        net = Unet3D_glob(feature_scale=arg.feature_scale, trans_feature=64)
        #net = Unet3D_glob2(feature_scale=arg.feature_scale, trans_feature=64)
        net = Unet3D_glob1106(feature_scale=arg.feature_scale, trans_feature=30)

    elif arg.model == "unet_reduced":
        net = UnetR3D(feature_scale=arg.feature_scale)
    else:
        raise NotImplementedError("Not Implemented Model")


    net = nn.DataParallel(net).to(torch_device)
    #if arg.loss == "l2":
    #    recon_loss = nn.L2Loss()
    #elif arg.loss == "l1":
    #    recon_loss = nn.L1Loss()

    ##for
    #backzero=1
    #recon_loss=FocalLoss3d_ver1(backzero=backzero)
    #val_loss=FocalLoss3d_ver1(gamma=2, pw=1, threshold=1.0, erode=3,backzero=backzero)

    recon_loss=DiceDis(ratio1=1, gamma=2, pw=1, erode=2, is_weight=0)
    val_loss=DiceDis(ratio1=0.1, is_weight=0)
    #val_loss=FocalLoss3d_ver2(is_weight=0)

    if(True):

        model = CNNTrainer(arg, net, torch_device, recon_loss=recon_loss, val_loss=val_loss, logger=logger)
        filename = "epoch[0207]_losssum[0.244572].pth.tar"
        model.load(filename=filename)
        test_loader = nucleusloader(f_path_test, batch_size=1, transform=None,
                                    cpus=arg.cpus, shuffle=True,
                                    drop_last=True)
        model.test(test_loader,savedir='hardcase'+filename)

        model = CNNTrainer(arg, net, torch_device, recon_loss=recon_loss, val_loss=val_loss, logger=logger)
        filename = "epoch[0208]_losssum[0.238254].pth.tar"
        model.load(filename=filename)
        test_loader = nucleusloader(f_path_test, batch_size=1, transform=None,
                                    cpus=arg.cpus, shuffle=True,
                                    drop_last=True)
        model.test(test_loader,savedir='hardcase'+filename)

    if(False):
        model = CNNTrainer(arg, net, torch_device, recon_loss=recon_loss, val_loss=val_loss, logger=logger)
        filename = "epoch[0324]_losssum[0.053456].pth.tar"
        model.load(filename=filename)
        test_loader = nucleusloader(f_path_valid, batch_size=1, transform=None,
                                    cpus=arg.cpus, shuffle=True,
                                    drop_last=True)
        model.test(test_loader,savedir='validset'+filename)


    if (False):
        model = BigTrainer(arg, net, torch_device, recon_loss=recon_loss, val_loss=val_loss, logger=logger)
        filename = "epoch[0418]_losssum[0.049294].pth.tar"
        model.load(filename=filename)

        f_path_hela = "/data1/20181018_TwoCell_Soomin"
        test_loader = Helaloader(f_path_hela, batch_size=1, transform=None,
                                    cpus=arg.cpus, shuffle=True,
                                    drop_last=True)
        model.test(test_loader, savedir='twoCell' + filename)