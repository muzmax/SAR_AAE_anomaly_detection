import torch.utils.data as data
import torch

class EmptyDataset(data.Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, item: int):
        assert False, "This code is unreachable"