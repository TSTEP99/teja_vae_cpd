"""
Implementation of a VAE based CP decomposition where we used the values along the spatial dimension 
to encoder epoch/sample values of our factor matrixs while other values are calculated in the same
way as https://arxiv.org/pdf/1611.00866.pdf 
"""
from teja_decoder_cpd import teja_decoder_cpd
from teja_encoder_cpd import teja_encoder_cpd
import torch.nn as nn

class teja_vae_cpd(nn.Module):
    def __init__(self, other_dims, output_channels = 32, kernel_size = 19, stride = 1, encoder_hidden_layer_size = 100, decoder_hidden_layer_size = 100, rank = 3, device = None):
        """Initializes the parameters and layer for Teja VAE"""

        #Calls constructor of super class
        super(teja_vae_cpd, self).__init__()

        #Initialize Encoder
        self.encoder = teja_encoder_cpd(other_dims, output_channels, kernel_size, stride, encoder_hidden_layer_size, rank, device).to(device = device)

        #Initialize Decoder
        self.decoder = teja_decoder_cpd(other_dims, decoder_hidden_layer_size, rank, device).to(device = device)


    def forward(self, x):
        """Does the forward pass for Teja VAE"""

        #Passes x through encoder of VAE
        means, log_vars, predicted_labels = self.encoder(x)

        #Passes mean and log variance throught decoder to try to reconstruct original tensor
        tensor = self.decoder(means, log_vars)

        return tensor, means, log_vars, predicted_labels
