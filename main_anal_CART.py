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
from datas.NucleusLoader import nucleusloader
from trainers.CNNTrainer import CNNTrainer
import copyreg
from models.unet_glob1111 import Unet3D_glob1111

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
    #arg.save_dir = "net_0912_absloss_F_HE_pw10_threshold1_erode3"
    arg.save_dir = "nets_1013_unet_glob1007_disdice_FRE_pw1_erode2_feat1_trans64"
    #arg.save_dir = "nets_1112_unet_glob1111_class_FRE_pw50_erode2_feat1_trans24_signdiff_newdata"
    #arg.save_dir = "nets_1021_unet_glob1007_disdice_FRE_pw1_erode2_feat1_trans64_bothplus"
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
    f_path_valid = "/home/jysong/PyCharmProjects_JY/180907_3DcellSegmentation_regressionVer/data/valid_"
    #f_path_test = "/home/jysong/PyCharmProjects_JY/180907_3DcellSegmentation_regressionVer/data/test_"
    f_path_test = "/data1/Moosung_CART/For_analysis"
    #f_path_test = "/data1/Moosung_CART/For_analysis/dataset1/exp0_fullsequence"

    preprocess = preprocess.get_preprocess(arg.augment)


    if arg.model == "fusion":
        net = Fusionnet(arg.in_channel, arg.out_channel, arg.ngf, arg.clamp)
    elif arg.model == "unet":
        net = Unet3D(feature_scale=arg.feature_scale)
    elif arg.model == "unet_gh":
        ## "nets_1004_unet_glob_absloss_FRE_pw10_erode2_feat1_trans30"
        net = Unet3D_glob(feature_scale=arg.feature_scale, trans_feature=64)
        # net = Unet3D_glob2(feature_scale=arg.feature_scale, trans_feature=64)
        # net=Unet3D_glob1111(feature_scale=arg.feature_scale, trans_feature=24)

    elif arg.model == "unet_reduced":
        net = UnetR3D(feature_scale=arg.feature_scale)
    else:
        raise NotImplementedError("Not Implemented Model")


    net = nn.DataParallel(net).to(torch_device)
    #if arg.loss == "l2":
    #    recon_loss = nn.L2Loss()
    #elif arg.loss == "l1":
    #    recon_loss = nn.L1Loss()

    backzero=1
    recon_loss=FocalLoss3d_ver1(backzero=backzero)
    val_loss=FocalLoss3d_ver1(gamma=2, pw=1, threshold=1.0, erode=3,backzero=backzero)


    model = CNNTrainer(arg, net, torch_device, recon_loss=recon_loss, val_loss=val_loss, logger=logger)
    #model.load(filename="epoch[0402]_losssum[0.016887].pth.tar")
    ##1013 version, quite nice
    #arg.save_dir = "nets_1008_unet_glob1007_disdice_FRE_pw10_erode2_feat1_trans64"
    #model.load(filename="epoch[0483]_losssum[0.016087].pth.tar")
    ##1014 version
    #model.load(filename="epoch[0418]_losssum[0.049294].pth.tar")

    model.load(filename="epoch[0324]_losssum[0.053456].pth.tar")

    #filename = "epoch[0401]_losssum[0.059813].pth.tar"
    #model.load(filename=filename)

    sdir="/data1/0000Deployed_JY/181013_analCART"

    ######phase 1######
    if(False):
        sdir_=sdir+'/dataset1'
        if os.path.exists(sdir_) is False:
            os.mkdir(sdir_)
        test_loader = nucleusloader(f_path_test+'/dataset1/exp0_fullsequence', batch_size=1, transform=None,
                                    cpus=arg.cpus, shuffle=True,
                                   drop_last=True)
        model.test(test_loader,savedir=sdir_)
    ######phase 2######

    if(False):
        listdir=os.listdir(f_path_test+'/dataset2')
        for ld in listdir:
            if('20180713_20180713.152640.905.stimCD841BB-KwoP-019' not in ld):
            #if('20180714_20180714.134210.177.stimCD441BBPkd-KwoP-026' not in ld):
                continue

            print(ld)
            #sdir_=sdir+'/dataset2_'+ld
            sdir_ = sdir + '/' + filename+ld
            if os.path.exists(sdir_) is False:
                os.mkdir(sdir_)
            test_loader = nucleusloader(f_path_test+'/dataset2/'+ld, batch_size=1, transform=None,
                                        cpus=arg.cpus, shuffle=True,
                                        drop_last=True)
            model.test(test_loader,savedir=sdir_)

    if(False):
        listdir=os.listdir(f_path_test+'/dataset3')
        for ld in listdir:
            sdir_=sdir+'/dataset3_'+ld
            if os.path.exists(sdir_) is False:
                os.mkdir(sdir_)
            test_loader = nucleusloader(f_path_test+'/dataset3/'+ld, batch_size=1, transform=None,
                                        cpus=arg.cpus, shuffle=True,
                                        drop_last=True)
            model.test(test_loader,savedir=sdir_)

    if(False):
        sdir_=sdir+'/dataset5'
        if os.path.exists(sdir_) is False:
            os.mkdir(sdir_)
        test_loader = nucleusloader(f_path_test+'/dataset5', batch_size=1, transform=None,
                                    cpus=arg.cpus, shuffle=True,
                                   drop_last=True)
        model.test(test_loader,savedir=sdir_)

    if(True):
        sdir_=sdir+'/dataset6'
        if os.path.exists(sdir_) is False:
            os.mkdir(sdir_)
        test_loader = nucleusloader(f_path_test+'/dataset6', batch_size=1, transform=None,
                                    cpus=arg.cpus, shuffle=True,
                                   drop_last=True)
        model.test(test_loader,savedir=sdir_)
    ######phase 1######

#    test_loader = nucleusloader(f_path_test+'/dataset3/RIFL', batch_size=1, transform=None,
#                                cpus=arg.cpus, shuffle=False,
#                                drop_last=True)
#    model.test(test_loader,savedir='dataset3')
