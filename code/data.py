from torch.utils.data import Dataset

class TensorDataset(Dataset):
    def __init__(self, tensor, labels):
        self.tensor = tensor
        self.class_labels = labels
    
    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        return self.tensor[idx], self.class_labels[idx]