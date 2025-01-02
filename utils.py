import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

class UnpaddedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataloader(dataset_path, batch_size):
    dataset = torch.load(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def get_feature(encoder, dataloader, inf_seed=None):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if inf_seed:
        set_all_seeds(inf_seed)

    features = np.array([])
    labels = np.array([])
    with torch.no_grad():
        for sequence, label in dataloader:
            sequence = sequence.type(torch.FloatTensor).to(DEVICE)
            label = label.to(DEVICE)

            feature = encoder(sequence)
            feature = feature.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            features = np.vstack([features, feature]) if features.size else feature
            labels = np.hstack([labels, label]) if labels.size else label

    return features, labels


def compute_output(net, features, net2=None):
    output = net(features)
    if net2:
        output = (output + net2(features)) / 2
    return output


def predict(output, is_binary):
    if is_binary:
        return (output > 0.5).long()
    return output.data.max(1, keepdim=True)[1]