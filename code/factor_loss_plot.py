"""Function to plot out and analyze loss over multiple factors"""
from data import TensorDataset
from helper import list_mus_vars, reparameterization
from losses import laplacian_total_variation_loss, original_loss, total_variation_loss, supervised_laplacian_total_variation_loss
from math import floor
from preprocess import create_indices, process_eegs
from torchmetrics import MeanSquaredError
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, random_split
from teja_vae import teja_vae
from train import train_loop, test_loop
import matplotlib.pyplot as plt
import numpy as np
import torch

def factor_loss_plot(num_rank, device = "cuda:1", batch_size=16, learning_rate = 1e-3, epochs = 100, encoder_hidden_layer_size = 400, \
                      decoder_hidden_layer_size = 400):


    full_psds, _, _, _, _, grade, epi_dx, alz_dx, _, _, _, _ = process_eegs()

    pop_psds= full_psds[(epi_dx<0) & (alz_dx<0)]

    pop_psds = (pop_psds - torch.min(pop_psds))/(torch.max(pop_psds) - torch.min(pop_psds))

    an_labels = grade[(epi_dx<0) & (alz_dx<0)].to(torch.float32).to(device)

    dims = pop_psds.shape

    print("Dimensions of population tensor:", dims)

    pop_psds = pop_psds.to(device)
    pop_psds = pop_psds.to(torch.float32) 

    total_length = pop_psds.shape[0]

    train_length = floor(0.9 * total_length)
    val_length = floor( 0.5 * (total_length-train_length))
    test_length = total_length - train_length - val_length


    lengths = [train_length, val_length, test_length]

    dataset= TensorDataset(pop_psds, an_labels)

    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths, generator = torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    other_dims = dims[1:] 

    best_rmse_losses = []
    best_test_losses = []
    for rank in trange(1, num_rank+1):    
        print("Rank {}".format(rank))
        model = teja_vae(other_dims, encoder_hidden_layer_size = encoder_hidden_layer_size, \
                        decoder_hidden_layer_size = decoder_hidden_layer_size, rank = rank, device = device)

        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, verbose = True)

        train_losses = []
        test_losses = []
        mse_losses = []
        
        for t in range(epochs):
            #print(f"Epoch {t+1}\n-------------------------------")
            train_loss = train_loop(train_dataloader, model, supervised_laplacian_total_variation_loss, optimizer, scheduler, verbose = False)
            test_loss, mse_loss = test_loop(test_dataloader, model, supervised_laplacian_total_variation_loss, verbose = False)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            mse_losses.append(mse_loss)
            scheduler.step(mse_loss)

        # print("Done!")
        # print(f" Epoch best test loss {np.argmin(test_losses)+1}")
        best_rmse_losses.append(np.min(mse_losses))
        best_test_losses.append(np.min(test_losses))

        fig, ax = plt.subplots()

        # Plot the RMSE curve in blue
        ax.plot(best_rmse_losses, color='blue', label='RMSE')

        # Add a legend and axis labels
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')

        fig.savefig('../data/rmse_loss.png')

        fig, ax = plt.subplots()

        # Plot the RMSE curve in blue
        ax.plot(best_test_losses, color='red', label='test_loss')

        # Add a legend and axis labels
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')

        fig.savefig('../data/test_loss.png')

if __name__ == "__main__":
    factor_loss_plot(50)


