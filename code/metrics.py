import numpy as np
from sklearn.metrics import auc
import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torch.distributions import Normal
from torchattacks.attacks.fgsm import FGSM


def accuracy(model: nn.Module, batches):
    correct = 0
    total = 0
    for images, labels in batches:
        out = model(images)
        labels_hat = torch.argmax(out, dim=1)
        correct += torch.sum(labels == labels_hat).item()
        total += len(labels)
    acc = correct / (total * 1.0)
    return acc


def FGSM_accuracy_curve(model: nn.Module, batches, max_delta=0.015, steps=20):
    delta_arr = np.linspace(0, max_delta, steps)
    fgsm_accuracy_arr = []

    for delta in delta_arr:
        attack = FGSM(model, eps=delta)

        correct = 0
        total = 0
        for images, labels in batches:
            with torch.enable_grad():
                adv_images = attack(images, labels)
            out = model(adv_images)
            labels_hat = torch.argmax(out, dim=1)
            correct += torch.sum(labels == labels_hat).item()
            total += len(labels)
        fgsm_accuracy_arr.append(correct / (total * 1.0))

    auc_fgsm = auc(delta_arr, fgsm_accuracy_arr) / max_delta
    return delta_arr, fgsm_accuracy_arr, auc_fgsm


def Noise_accuracy_curve(model: nn.Module, batches, max_delta=0.08, steps=40):
    delta_arr = np.linspace(0, max_delta, steps)
    noise_accuracy_arr = []
    param_vector = parameters_to_vector(model.parameters())

    for delta in delta_arr:
        new_param_vector = param_vector.detach().clone()
        if delta > 1e-7:
            noise = (
                Normal(0, delta)
                .sample((len(new_param_vector),))
                .to(next(model.parameters()).device)
            )
            new_param_vector.add_(noise)
        vector_to_parameters(new_param_vector, model.parameters())

        correct = 0
        total = 0
        for images, labels in batches:
            out = model(images)
            labels_hat = torch.argmax(out, dim=1)
            correct += torch.sum(labels == labels_hat).item()
            total += len(labels)
        noise_accuracy_arr.append(correct / (total * 1.0))

    vector_to_parameters(param_vector, model.parameters())

    auc_noise = auc(delta_arr, noise_accuracy_arr) / max_delta
    return delta_arr, noise_accuracy_arr, auc_noise
