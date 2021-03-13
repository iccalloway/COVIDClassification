import os
import pandas as pd
import soundfile as sf
import torch

from collections import Counter
from torch.utils.data import Dataset

class COVIDDataset(Dataset):
    def __init__(self, path, grouping_variables):
        data=pd.read_csv(path)
        self.data = data
        self.path = os.path.dirname(path)
        classes = data.apply(lambda x:'_'.join(x[grouping_variables]), axis=1).to_frame(name='factor')
        classes['idx'] = classes.index
        self.classes = classes
        self.counts = Counter(self.classes['factor'])
        return

    def __getitem__(self, i):
        return self.data.iloc[i]

    def __len__(self):
        return self.data.shape[0]

    def collate_batch(self, batch):
        inputs = []
        labels = []
        for item in batch:
            audio, sr = sf.read(os.path.join(self.path, 'AUDIO', item['File_name']+".flac"))
            status = 1 if item['Covid_status'] == 'p' else 0
            inputs.append(audio)
            labels.append(status)
        return (inputs, torch.unsqueeze(torch.tensor(labels),0))


class DiCOVATrack2(Dataset):
    """
    Instantiate same as above object, but specify further which type of data
    is desired.
    """
    def __init__(self, path, grouping_variables, dataset_type):
        datasets = ['breathing-deep', 'counting-normal', 'vowel-e']
        if dataset_type not in datasets:
            raise NotImplementedError
        self.dataset_type = dataset_type
        data=pd.read_csv(path)
        self.fn_col_name = 'ID'
        data[self.fn_col_name] += f'_{self.dataset_type}'
        self.data = data
        self.path = os.path.dirname(path)
        classes = data.apply(lambda x:'_'.join(x[grouping_variables]), axis=1).to_frame(name='factor')
        classes['idx'] = classes.index
        self.classes = classes
        self.counts = Counter(self.classes['factor'])
        return

    def __getitem__(self, i):
        return self.data.iloc[i]

    def __len__(self):
        return self.data.shape[0]

    def collate_batch(self, batch):
        inputs = []
        labels = []
        for item in batch:
            fn = item[self.fn_col_name]
            audio, sr = sf.read(os.path.join(self.path, 'AUDIO', self.dataset_type,
                f"{fn}.flac"))
            status = 1 if item['Covid_status'] == 'p' else 0
            inputs.append(audio)
            labels.append(status)
        return (inputs, torch.unsqueeze(torch.tensor(labels),0))
