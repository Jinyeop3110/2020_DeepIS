import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import copy

# https://github.com/GunhoChoi/Grad-CAM-Pytorch
class GuidedReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[grad_input < 0] = 0
        grad_input[input < 0]      = 0
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, device):
        self.device = device
        self.model = copy.deepcopy(model)
        self.model.eval()
        guidedReLU = GuidedReLU.apply
        
        def _get_find_seq(sequential):
            for idx, module in sequential._modules.items():
                if module.__class__ == nn.ReLU:
                    sequential._modules[idx] = guidedReLU
                _get_find_seq(module)
                    
        _get_find_seq(self.model)
                
    def __str__(self):
        tmp = ""
        for name, module in self.model._modules.items():
            tmp += name + " " + str(module) + "\n"
        return tmp
    
    def __repr__(self):
        return self.__str__()
            
    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        output = self.forward(input.to(self.device))
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).to(self.device)
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot * output)
        
        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().numpy()
        output = output[0,:,:,:]

        return output.transpose(1, 2, 0)[:, :, 0]
    
    @staticmethod
    def gb_on_cam(mask, gb_img):
        return np.multiply(mask, gb_img)
