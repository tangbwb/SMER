import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

class Weight_Selection(nn.Module):
    def __init__(self, weight_len) -> None:
        super(Weight_Selection,self).__init__()
        self.weight = nn.parameter.Parameter(torch.ones([weight_len]))

    def forward(self, x, index):
        x = self.weight[index] * x
        return x

def MI_FGSM_SMER(surrogate_models,images, labels, args,num_iter = 10):
    eps = args.eps/255.0
    alpha = args.alpha/255.0
    beta = alpha
    momentum = args.momentum
    image_min = clip_by_tensor(images - eps, 0.0, 1.0)
    image_max = clip_by_tensor(images + eps, 0.0, 1.0)
    m = len(surrogate_models) 
    m_smer = m*4
    weight_selection = Weight_Selection(m).to(images.device)
    optimizer = torch.optim.SGD(weight_selection.parameters(),lr=2e-2,weight_decay=2e-3)
    noise = 0
    grad = 0
    for i in range(num_iter):
        if images.grad is not None:
            images.grad.zero_()
        images = Variable(images, requires_grad = True)
        x_inner = images.detach()
        x_before = images.clone()
        noise_inner_all = torch.zeros([m_smer, *images.shape]).to(images.device)
        grad_inner = torch.zeros_like(images)
        options = []
        for i in range(int(m_smer / m)):
            options_single=[j for j in range(m)]
            np.random.shuffle(options_single)
            options.append(options_single)
        options = np.reshape(options,-1)
        for j in range(m_smer):
            option = options[j]
            grad_single = surrogate_models[option]
            x_inner.requires_grad = True
            out_logits = grad_single(x_inner)
            if type(out_logits) is list:
                out = weight_selection(out_logits[0],option)
                aux_out = weight_selection(out_logits[1],option)
            else:
                out = weight_selection(out_logits,option)
            loss = F.cross_entropy(out, labels)
            if type(out_logits) is list:
                loss = loss + F.cross_entropy(aux_out, labels)
            noise_im_inner = torch.autograd.grad(loss,x_inner)[0]
            group_logits = 0
            group_aux_logits = 0
            for m_step, model_s in enumerate(surrogate_models):
                out_logits = model_s(x_inner)
                if type(out_logits) is list:
                    logits = weight_selection(out_logits[0],m_step)
                    aux_logits = weight_selection(out_logits[1],m_step)
                else:
                    logits = weight_selection(out_logits,m_step)
                group_logits = group_logits + logits / m
                if type(out_logits) is list:
                    group_aux_logits = group_aux_logits + aux_logits / m
            loss = F.cross_entropy(group_logits,labels)
            if type(out_logits) is list:
                loss = loss + F.cross_entropy(group_aux_logits,labels)
            outer_loss = -torch.log(loss)
            x_inner.requires_grad = False
            outer_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            noise_inner = noise_im_inner
            noise_inner = noise_inner / torch.mean(torch.abs(noise_inner), dim=[1, 2, 3], keepdims=True)
            grad_inner = grad_inner + noise_inner
            x_inner = x_inner + beta * torch.sign(grad_inner)
            x_inner = clip_by_tensor(x_inner, image_min, image_max)
            noise_inner_all[j] = grad_inner.clone()
        noise =noise_inner_all[-1].clone() 
        noise = noise / torch.mean(torch.abs(noise), dim=[1, 2, 3], keepdims=True)
        grad = noise + momentum * grad
        images = x_before +  alpha * torch.sign(grad)
        images = clip_by_tensor(images, image_min, image_max)
    return images

def I_FGSM_SMER(surrogate_models,images, labels, args,num_iter = 10):
    eps = args.eps/255.0
    alpha = args.alpha/255.0
    beta = alpha
    image_min = clip_by_tensor(images - eps, 0.0, 1.0)
    image_max = clip_by_tensor(images + eps, 0.0, 1.0)
    m = len(surrogate_models) 
    m_smer = m*4
    weight_selection = Weight_Selection(m).to(images.device)
    optimizer = torch.optim.SGD(weight_selection.parameters(),lr=2e-2,weight_decay=2e-3)
    grad = 0
    for _ in range(num_iter):
        if images.grad is not None:
            images.grad.zero_()
        images = Variable(images, requires_grad = True)
        x_inner = images.detach()
        x_before = images.clone()
        noise_inner_all = torch.zeros([m_smer, *images.shape]).to(images.device)
        grad_inner = torch.zeros_like(images)
        options = []
        for _ in range(int(m_smer / m)):
            options_single=[j for j in range(m)]
            np.random.shuffle(options_single)
            options.append(options_single)
        options = np.reshape(options,-1)
        for j in range(m_smer):
            option = options[j]
            grad_single = surrogate_models[option]
            x_inner.requires_grad = True
            out_logits = grad_single(x_inner)
            if type(out_logits) is list:
                out = weight_selection(out_logits[0],option)
                aux_out = weight_selection(out_logits[1],option)
            else:
                out = weight_selection(out_logits,option)
            loss = F.cross_entropy(out, labels)
            if type(out_logits) is list:
                loss = loss + F.cross_entropy(aux_out, labels)
            noise_im_inner = torch.autograd.grad(loss,x_inner)[0]
            group_logits = 0
            group_aux_logits = 0
            for m_step, model_s in enumerate(surrogate_models):
                out_logits = model_s(x_inner)
                if type(out_logits) is list:
                    logits = weight_selection(out_logits[0],m_step)
                    aux_logits = weight_selection(out_logits[1],m_step)
                else:
                    logits = weight_selection(out_logits,m_step)
                group_logits = group_logits + logits / m
                if type(out_logits) is list:
                    group_aux_logits = group_aux_logits + aux_logits / m
            loss = F.cross_entropy(group_logits,labels)
            if type(out_logits) is list:
                loss = loss + F.cross_entropy(group_aux_logits,labels)
            outer_loss = -torch.log(loss)
            x_inner.requires_grad = False
            outer_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            noise_inner = noise_im_inner
            noise_inner = noise_inner / torch.mean(torch.abs(noise_inner), dim=[1, 2, 3], keepdims=True)
            grad_inner = grad_inner + noise_inner
            x_inner = x_inner + beta * torch.sign(grad_inner)
            x_inner = clip_by_tensor(x_inner, image_min, image_max)
            noise_inner_all[j] = grad_inner.clone()
        grad =noise_inner_all[-1].clone()
        images = x_before +  alpha * torch.sign(grad)
        images = clip_by_tensor(images, image_min, image_max)
    return images
