from train import *
from preprocess import create_indices, process_eegs
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    ENCODER_HIDDEN_LAYER_SIZE = 400
    DECODER_HIDDEN_LAYER_SIZE = 400

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

    best_mse_losses = []

    for rank in range(2, 101): 

        model = teja_vae_cpd(other_dims, encoder_hidden_layer_size = ENCODER_HIDDEN_LAYER_SIZE, decoder_hidden_layer_size = DECODER_HIDDEN_LAYER_SIZE\
                            , rank = rank, device = DEVICE)

        model.to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, betas = (0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3, verbose = True)

        train_losses = []
        test_losses = []
        mse_losses = []

        for t in range(EPOCHS):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = train_loop(train_dataloader, model, supervised_original_loss, optimizer, scheduler, verbose = False)
            test_loss, mse_loss = test_loop(test_dataloader, model, supervised_original_loss)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            mse_losses.append(mse_loss)
            scheduler.step(test_loss)
        
        best_mse_losses.append(np.min(mse_losses))
        plt.plot(np.arange(2,101),best_mse_losses)
        plt.savefig("../data/reconstruction_plot.png")
        # print("Done!")
        # print(f" Epoch best test loss {np.argmin(test_losses)+1}")