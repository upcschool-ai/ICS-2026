import torch
import pandas as pd
from PIL import Image


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, images_path: str, labels_path: str):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int):
        pass