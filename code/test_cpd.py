"""
File to test the functionality of all implemented modules in Teja-VAE
Note: These tests are now invalid will need to update in the future
"""
from teja_encoder_cpd import teja_encoder_cpd
from teja_decoder_cpd import teja_decoder_cpd
from teja_vae_cpd import teja_vae_cpd
import torch

def test_encoder_cpd():
    """Function to test the teja encoder module"""

    sample_tensor = torch.randn(14052, 19, 45)
    other_dims = sample_tensor.shape[1:]
    encoder = teja_encoder_cpd(other_dims = other_dims)
    means, log_vars, predicted_labels = encoder(sample_tensor)

    assert tuple(means.shape) == (14052, 3)
    assert tuple(log_vars.shape) == (14052, 3)
    assert tuple(predicted_labels.shape) == (14052, 1)

def test_decoder_cpd():
    """Function to test the teja decoder module"""

    means = torch.randn((14052,3))
    log_vars = torch.randn((14052,3))
    other_dims = [19, 45]

    decoder = teja_decoder_cpd(other_dims = other_dims)
    tensor = decoder(means, log_vars)

    assert tuple(tensor.shape) == (14052, 19, 45)

def test_vae_cpd():
    """Function to test Teja-VAE"""

    device = "cpu"
    sample_tensor = torch.randn(14052, 19, 45, device = device)
    other_dims = sample_tensor.shape[1:]
    vae = teja_vae_cpd(other_dims = other_dims, device = device)
    tensor, means, log_vars, predicted_labels = vae(sample_tensor)

    assert sample_tensor.shape == tensor.shape
    assert means.shape[0] == sample_tensor.shape[0]
    assert means.shape[1] == 3
    assert log_vars.shape[0] == sample_tensor.shape[0]
    assert log_vars.shape[1] == 3
    assert predicted_labels.shape[0] == sample_tensor.shape[0]
    assert predicted_labels.shape[1] == 1

if __name__ == "__main__":
    test_encoder_cpd()
    test_decoder_cpd()
    test_vae_cpd()
