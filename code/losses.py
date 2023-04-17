"""File containing the losses for Teja-VAE"""
from torchmetrics import MeanSquaredError
import numpy as np
import torch

def compute_laplacian_loss(mu):
    distance_matrix = np.load("../data/distance_matrix.npy")
    distance_matrix = torch.from_numpy(distance_matrix).to(mu.device).to(mu.dtype)
    avg_mu = distance_matrix @ mu

    return torch.pow(mu - avg_mu, 2).mean()

def compute_total_variation_loss(factor):

    return ((factor[1:] - factor[:-1]).pow(2)).mean()

def reconstruction_loss(samples, mean, var):
    """Computes the reconstruction loss in similair fashion to VAE CP"""
    #print(mean.shape, log_var.shape, samples.shape)
    L = samples.shape[0]
    std = torch.sqrt(var)

    return (-torch.log(std) - 0.5 * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * ((samples - mean)/std) ** 2 ).mean()#.sum()/L

# def reconstruction_loss(samples, preds):
#     mean_squared_error = MeanSquaredError(squared = True).to(preds.device)
#     return mean_squared_error(preds, samples).mean()

def regularization_loss(mus, lambdas, mus_tildes, lambdas_tildes):
    """Computes the regularization loss in a similair fashion to the VAE-CP paper"""
    loss = 0
    for i in range(len(mus)):
        mu = mus[i]
        lambda_ = lambdas[i]
        mu_tilde = mus_tildes[i]
        lambda_tilde = lambdas_tildes[i]

        var = torch.exp(lambda_)
        var_tilde = torch.exp(lambda_tilde)
        std = torch.sqrt(var)
        std_tilde = torch.sqrt(var_tilde)
        loss += (torch.log(std_tilde/std) + (var + (mu - mu_tilde)**2)/(2 * var_tilde) - 0.5).mean()#.sum()
    return loss

def original_loss(samples, mean, log_var, mus, lambdas, mus_tildes, lambdas_tildes, predicted_labels = None, target_labels = None, dims = None):
    """Computes a loss function very similair to the VAE-CP paper"""
    rec_loss = reconstruction_loss(samples, mean, log_var)
    reg_loss = regularization_loss(mus, lambdas, mus_tildes, lambdas_tildes) 
    return -rec_loss + reg_loss

def total_variation_loss(elements, preds, mus, lambdas, mus_tildes, lambdas_tildes, beta = 10 ,dims = [2]):
    """The loss function with a total variation term added"""
    loss = original_loss(elements, preds, mus, lambdas, mus_tildes, lambdas_tildes)

    for i in range(len(mus)):
        if i in dims:
            loss += compute_total_variation_loss(mus[i])

    return loss


def supervised_original_loss(elements, mean, log_var, mus, lambdas, mus_tildes, lambdas_tildes, predicted_labels, target_labels, dims = [2]):
    bce_loss = torch.nn.BCELoss()
    return original_loss(elements,  mean, log_var, mus, lambdas, mus_tildes, lambdas_tildes, dims) \
    + 1* bce_loss(predicted_labels, target_labels.view(-1,1))

def laplacian_total_variation_loss(elements,preds, mus, lambdas, mus_tildes, lambdas_tildes, dims = [2], spatial_dim=1):

    return total_variation_loss(elements, preds, mus, lambdas, mus_tildes, lambdas_tildes, dims) + compute_laplacian_loss(mus[spatial_dim])

def supervised_total_variation_loss(elements, preds, mus, lambdas, mus_tildes, lambdas_tildes, predicted_labels, target_labels, dims = [2]):
    bce_loss = torch.nn.BCELoss()
    return total_variation_loss(elements, preds, mus, lambdas, mus_tildes, lambdas_tildes, dims) + bce_loss(predicted_labels, target_labels.view(-1,1))

def supervised_laplacian_total_variation_loss(elements, preds, mus, lambdas, mus_tildes, lambdas_tildes, predicted_labels, target_labels, dims = [2]):
    bce_loss = torch.nn.BCELoss()
    return laplacian_total_variation_loss(elements, preds, mus, lambdas, mus_tildes, lambdas_tildes, dims) \
    + bce_loss(predicted_labels, target_labels.view(-1,1))
