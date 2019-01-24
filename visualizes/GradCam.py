import torch
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
from GuidedBackpropReLUModel import GuidedBackpropReLUModel
import copy

class GradCam:
    def __init__(self, model, feature_layer, device):
        self.device = device
        self.model = copy.deepcopy(model)        

        self.feature = []
        self.grad = []
        
        hook_layer = self.model._modules[feature_layer[0]]._modules[feature_layer[1]]
        self.hook = hook_layer.register_backward_hook(self.save_grad)
        self.feature_hook = hook_layer.register_forward_hook(self.save_feature)
    
    def save_feature(self, layer, input_, output):
        self.feature.append(output)
    
    def save_grad(self, layer, grad_in, grad_out):
        self.grad.append(grad_out[0])
        
    def remove_hook_data(self):
        self.feature = []
        self.grad = []
        
    def forward(self, input_):
        return self.model(input_) 

    def __call__(self, input_, index=None, mode="gdcampp"):
        output = self.model(input_.to(self.device))
        
        if index is None:
            index = np.argmax(output.cpu().detach().numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).to(self.device)
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        # Capital A
        output_np   = output.cpu().detach().numpy()[0, :]
        output_relu = np.maximum(output_np, 0)
        
        grads_val = self.grad[0].cpu().numpy()
        target = self.feature[0].cpu().detach().numpy()[0, :]
                                
        if mode in ["gdcam", "gdcampp"]:
            if mode == "gdcampp":
               # grad_cam ++
                grad_2 = grads_val * grads_val
                grad_3 = grads_val * grad_2
                grads_val = grad_2 / (2 * grad_2 + grad_3 * target + 1e-12)
            weights = np.mean(grads_val, axis = (2, 3))[0, :] # normal grad_cam
            
            cam = np.zeros(target.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]

        elif mode == "gunho":
            grad = self.grad[0]
            alpha = torch.sum(grad, dim=3, keepdim=True)
            alpha = torch.sum(alpha,     dim=2, keepdim=True)

            cam = alpha[0] * grad[0]
            cam = torch.sum(cam, dim=0).cpu().numpy()

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        self.remove_hook_data()
        return cam
    
    @staticmethod 
    def plotting_cam_on_image(img, mask, plt, subplot=111):
        def transparent_cmap(cmap, N=255):
            "Copy colormap and set alpha values"

            mycmap = cmap
            mycmap._init()
            mycmap._lut[:,-1] = np.linspace(0.5, 0.8, N+4)
            return mycmap

        mycmap = transparent_cmap(plt.cm.nipy_spectral)
        plt.subplot(subplot)
        plt.imshow(img[:, :, 0])
        plt.imshow(mask, cmap=mycmap, interpolation="nearest")
    
    @staticmethod
    def cam_on_image(img, mask):
        # TODO: FIX!!
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def _preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input_ = preprocessed_img
    input_.requires_grad_(True)
    return input_
    

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    # Can work with any model, but it assumes that the model has a 
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    device = torch.device("cuda")
    model = models.vgg19(pretrained=True).to(device)
    grad_cam = GradCam(model=model, target_layer_names = ["35"], device=device)
    gb_model = GuidedBackpropReLUModel(model=model, device=device)
    exit()
    img_path = None
    img = cv2.imread(img_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input_ = _preprocess_image(img)

    target_index = None

    cam_mask = grad_cam(input_, target_index)
    # cam_img is cam heatmap on orignal image.
    cam_img = GradCam.cam_on_image(img, cam_mask)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    gb_img = gb_model(input_, index=target_index)

    gb_mask = np.zeros(gb_img.shape)
    for i in range(0, gb_img.shape[0]):
        gb_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb_img)
    
    utils.save_image(cam_img, "cam.jpg")
    utils.save_image(torch.from_numpy(gb_img), 'gb.jpg')
    utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')