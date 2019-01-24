import random
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, rotate, map_coordinates
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import scipy.misc
from glob import glob
from scipy import io

def random_crop3d(input_, target_):
    if (np.random.random() > 0.4):
        return input_, target_
    zoom_rates = [1.1, 1.2, 1.3]
    zoom_rate  = random.choice(zoom_rates)


    zoom_input  = zoom(input_,  zoom_rate)
    zoom_target = zoom_rate * zoom(target_, zoom_rate)

    zoom_shape, img_shape = zoom_input.shape, input_.shape
    dx = random.randint(0, zoom_shape[0] - img_shape[0])
    dy = random.randint(0, zoom_shape[1] - img_shape[1])
    dz = random.randint(0, zoom_shape[2] - img_shape[2])

    zoom_input  = zoom_input[dx:dx + img_shape[0], dy:dy + img_shape[1], dz:dz + img_shape[2] ]
    zoom_target = zoom_target[dx:dx + img_shape[0], dy:dy + img_shape[1], dz:dz + img_shape[2] ]

    return zoom_input, zoom_target

def random_flip(input_, target_):
    flip = random.randint(0, 7) # 0, 1, 2

    if   flip == 0:
        return input_[:, ::-1, ::-1],    target_[:, ::-1, ::-1]
    elif flip == 1:
        return input_[:, ::-1, :],    target_[:, ::-1, :]
    elif flip == 2:
        return input_[:, :, ::-1], target_[:, :, ::-1]
    elif flip == 3:
        return input_, target_
    elif flip == 4:
        return input_[::-1, ::-1, ::-1], target_[::-1, ::-1, ::-1]
    elif flip == 5:
        return input_[::-1, ::-1, :], target_[::-1, ::-1, :]
    elif flip == 6:
        return input_[::-1, :, ::-1], target_[::-1, :, ::-1]
    elif flip == 7:
        return input_[::-1, :, :], target_[::-1, :, :]

def random_rotate(input_, target_):
    angle=random.randint(10, 350)

    rotate_input  = rotate(input_, angle,
                           reshape=False)
    rotate_target = rotate(target_, angle,
                           reshape=False)
    return rotate_input, rotate_target


def elastic_transform(input_, target_, weight_=None, param_list=None, random_state=None):
    if(np.random.random()>0.5):
        return input_, target_

    if param_list is None:
        ##HHE
        #param_list = [(0, 1),(2, 1), (5, 2), (4, 3)]

        ##HHE
        param_list = [(2, 2),(3,1)]

        ##E
        #param_list = [(1,0.5)]


    alpha, sigma = random.choice(param_list)

    assert len(input_.shape)==3
    shape = input_.shape

    if random_state is None:
       random_state = np.random.RandomState(None)    

    dx = gaussian_filter((random_state.rand(*shape[0:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape[0:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    #print(np.mean(dx), np.std(dx), np.min(dx), np.max(dx))

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    transformed = []
    for image in [input_, target_]:
        new = np.zeros(shape)
        if len(shape) == 3:
            for i in range(image.shape[2]):
                new[:, :, i] = map_coordinates(image[:, :, i], indices, order=1, mode="reflect").reshape(shape[0:2])
        else:
            new[:, :] = map_coordinates(image[:, :], indices, order=1, mode="reflect").reshape(shape)
        transformed.append(new)

    return transformed


ARG_TO_DICT = {
        "crop":random_crop3d,
        "flip":random_flip,
        "elastic":elastic_transform,
        "rotate":random_rotate
        }

def get_preprocess(preprocess_list):
    if not preprocess_list:
        return []
    return [ARG_TO_DICT[p] for p in preprocess_list.split(",")]


if __name__ == "__main__":
    img_path="/data1/0000Deployed_JY/strain_/20180520_20180520.160331.532.stomCD4PKD-KwoP-100_000000_002_.mat"
    data_pack = io.loadmat(img_path)
    # 2D ( 1 x H x W )
    input_np = data_pack['input']
    target_np = data_pack['target']

    input = input_np
    target= target_np

    data = {}
    data['input']=input
    data['target'] = target

    input_e,target_e = elastic_transform(input, target)
    data['input_e'] = input_e
    data['target_e'] = target_e

    input_c,target_c = random_crop3d(input, target)
    data['input_c'] = input_c
    data['target_c'] = target_c

    input_r,target_r = random_rotate(input, target)
    data['input_r'] = input_r
    data['target_r'] = target_r

    scipy.io.savemat("/home/jysong/PyCharmProjects_JY/181007_3DcellSegmentation_regressionVer" + "/preprocesstest.mat", data)


