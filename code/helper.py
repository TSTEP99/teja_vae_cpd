"""File defining some of the helper functions associated with Teja-VAE"""
import torch

def list_mus_vars(latent_means, latent_log_vars, model):
    mus_list = [latent_means]
    lambdas_list = [latent_log_vars]
    mus_tildes_list = [model.decoder.original_mu]
    lambdas_tildes_list = [model.decoder.original_lambda]

    mus_list.extend(model.decoder.other_mus)
    lambdas_list.extend(model.decoder.other_lambdas)
    mus_tildes_list.extend(model.decoder.other_mus_tildes)
    lambdas_tildes_list.extend(model.decoder.other_lambdas_tildes)

    return mus_list, lambdas_list, mus_tildes_list, lambdas_tildes_list


def reparameterization(mean, log_var):
    """Uses the parameterization trick from the original VAE formulation"""

    epsilons = torch.randn_like(log_var, device = mean.device)

    return mean + epsilons * torch.sqrt(torch.exp(log_var))
