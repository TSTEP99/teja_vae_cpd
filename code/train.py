"""File used to train VAE-CP"""
from data import TensorDataset
from helper import list_mus_vars, reparameterization
from losses import laplacian_total_variation_loss, original_loss, total_variation_loss, supervised_laplacian_total_variation_loss, supervised_total_variation_loss
from math import floor
from preprocess import create_indices, process_eegs
from torchmetrics import MeanSquaredError
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from teja_vae_cpd import teja_vae_cpd
import numpy as np
import torch

def train_loop(dataloader, model, loss_fn, optimizer, scheduler, verbose = True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    model.train()

    for batch, (samples, target_labels) in tqdm(enumerate(dataloader)):
        # Compute prediction and loss
        tensor, latent_means, latent_log_vars, predicted_labels = model(samples)
        mus, lambdas, mus_tildes, lambdas_tildes = list_mus_vars(latent_means, latent_log_vars, model)
        loss = loss_fn(samples, tensor, mus, lambdas, mus_tildes, lambdas_tildes, predicted_labels, target_labels)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if verbose and batch % 100 == 0:
            print(f"train loss at batch {batch}: {loss}")

    train_loss /= num_batches
    if verbose:
        print("Train Loss:", train_loss)

    return train_loss

def test_loop(dataloader, model, loss_fn, verbose = True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, mse_loss = 0, 0
    model.eval()


    with torch.no_grad():
        for batch, (samples, target_labels) in tqdm(enumerate(dataloader)):
            tensor, latent_means, latent_log_vars, predicted_labels = model(samples)
            mus, lambdas, mus_tildes, lambdas_tildes = list_mus_vars(latent_means, latent_log_vars, model)
            loss = loss_fn(samples, tensor, mus, lambdas, mus_tildes, lambdas_tildes, predicted_labels, target_labels)
            test_loss += loss.item()
            mean_squared_error = MeanSquaredError(squared = False).to(samples.device)
            mse_loss +=  mean_squared_error(samples, tensor).item()

    test_loss /= num_batches
    mse_loss /= num_batches
    if verbose:
        print("Test Loss:", test_loss)
        print("Test RMSE:", mse_loss)
    return test_loss, mse_loss

if __name__ == "__main__":
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    ENCODER_HIDDEN_LAYER_SIZE = 400
    DECODER_HIDDEN_LAYER_SIZE = 400
    RANK = 6

    full_psds, _, _, _, _, grade, epi_dx, alz_dx, _, _, _, _ = process_eegs()

    pop_psds= full_psds[(epi_dx<0) & (alz_dx<0)]

    pop_psds = (pop_psds - torch.min(pop_psds))/(torch.max(pop_psds) - torch.min(pop_psds))

    an_labels = grade[(epi_dx<0) & (alz_dx<0)].to(torch.float32).to(DEVICE)

    dims = pop_psds.shape

    print("Dimensions of population tensor:", dims)

    pop_psds = pop_psds.to(DEVICE)
    pop_psds = pop_psds.to(torch.float32) 

    total_length = pop_psds.shape[0]

    train_length = floor(0.9 * total_length)
    val_length = floor( 0.5 * (total_length-train_length))
    test_length = total_length - train_length - val_length


    lengths = [train_length, val_length, test_length]

    dataset= TensorDataset(pop_psds, an_labels)

    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths, generator = torch.Generator().manual_seed(42))

    print(f"Training Set has length {train_dataset.__len__()}")
    print(f"Validation Set has length {val_dataset.__len__()}")
    print(f"Test Set has length {test_dataset.__len__()}")

    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

    other_dims = dims[1:] 

    model = teja_vae_cpd(other_dims, encoder_hidden_layer_size = ENCODER_HIDDEN_LAYER_SIZE, decoder_hidden_layer_size = DECODER_HIDDEN_LAYER_SIZE\
                         , rank = RANK, device = DEVICE)

    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, betas = (0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, verbose = True)

    train_losses = []
    test_losses = []
    mse_losses = []
    
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(train_dataloader, model, supervised_total_variation_loss, optimizer, scheduler)
        test_loss, mse_loss = test_loop(test_dataloader, model, supervised_total_variation_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        mse_losses.append(mse_loss)
        torch.save(model, f'../checkpoints/teja_vae_cp_epoch_{t+1}.pth')
        scheduler.step(test_loss)

    print("Done!")
    print(f" Epoch best test loss {np.argmin(test_losses)+1}")
