import os
import pandas as pd
import soundfile as sf
import torch

from collections import Counter
from torch.utils.data import Dataset
import numpy as np

#import torchaudio
import torchvision
from PIL import Image
from datasetaug import *
import tensorflow as tf

class COVIDDataset(Dataset):
    def __init__(self, path, grouping_variables, cnn = False, data_aug = 0):
        data=pd.read_csv(path)
        self.data = data
        self.path = os.path.dirname(path)
        if len(grouping_variables) > 0:
            classes = data.apply(lambda x:'_'.join(x[grouping_variables]), axis=1).to_frame(name='factor')
            classes['idx'] = classes.index
            self.classes = classes
            self.counts = Counter(self.classes['factor'])
        self.cnn = cnn
        self.data_aug = data_aug
        return

    def __getitem__(self, i):
        return self.data.iloc[i]

    def __len__(self):
        return self.data.shape[0]

    def collate_batch(self, batch):
        inputs = []
        transforms = [None, MelSpectrogram(128, mode='train'), FrequencyMask(), TimeMask(), Volume()]
        labels = []
        for item in batch:
            audio, sr = sf.read(os.path.join(self.path, 'AUDIO', item['File_name']+".flac"))
            status = 1 if item['Covid_status'] == 'p' else 0
            # process spectrogram
            if self.cnn:    
                aug = np.random.choice([0, 4])         #[0] + [self.data_aug]           
                if aug == 1:
                    audio = transforms[aug](audio)
                elif aug > 0:
                    audio = extract_spectrogram(audio)
                    audio = transforms[aug](audio)
                else:
                    audio = extract_spectrogram(audio)
            inputs.append(audio if not self.cnn else torch.tensor(audio))
            labels.append(status)
        if self.cnn:
            return (torch.stack(inputs), torch.unsqueeze(torch.tensor(labels),0))
        return inputs, torch.unsqueeze(torch.tensor(labels),0)

class COVIDTest(Dataset):
    def __init__(self, path, grouping_variables):
        data = pd.read_csv(path)
        self.data = data
        self.path = os.path.dirname(path)
        if len(grouping_variables) > 0:
            classes = data.apply(
                lambda x: "_".join(x[grouping_variables]), axis=1
            ).to_frame(name="factor")
            classes["idx"] = classes.index
            self.classes = classes
            self.counts = Counter(self.classes["factor"])
        return

    def __getitem__(self, i):
        return self.data.iloc[i]

    def __len__(self):
        return self.data.shape[0]

    def collate_batch(self, batch):
        inputs = []
        #transforms = MelSpectrogram(128, mode='train')
        labels = []
        for item in batch:
           audio, sr = sf.read(
                os.path.join(self.path, "AUDIO", item["File_name"] + ".flac")
            )[:200000]
           status = 1 if item["Covid_status"] == "p" else 0
           inputs.append(audio)
           labels.append(status)
        return (inputs, item["File_name"], torch.unsqueeze(torch.tensor(labels), 0))

def extract_spectrogram(audio):
    sampling_rate = 44100
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]
    centre_sec = 2.5
    #3x128x250
    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[i]*sampling_rate/1000))
        hop_length = int(round(hop_sizes[i]*sampling_rate/1000))

        clip = torch.Tensor(audio)
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_fft=4410, win_length=window_length, hop_length=hop_length, n_mels=128)(clip)
        eps = 1e-6
        spec = spec.numpy()
        spec = np.log(spec+ eps)
        spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
        specs.append(spec)
    return torch.tensor(np.array(specs))


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


class FSDTrainDataset(Dataset):
    def __init__(self, path, grouping_variables=['label', 'manually_verified']):
        data=pd.read_csv(path)
        data['manually_verified'] = data['manually_verified'].astype(str)
        self.data = data
        self.path = os.path.dirname(path).replace('meta','audio_train')
        classes = data.apply(lambda x:'_'.join(x[grouping_variables]), axis=1).to_frame(name='factor')
        self.id2label = {k:v for k,v in enumerate(sorted(self.data['label'].unique()))}
        self.label2id = {v:k for k,v in self.id2label.items()}
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
            audio, sr = sf.read(os.path.join(self.path, item['fname']))
            inputs.append(audio)
            labels.append(self.label2id[item['label']])
        return (inputs, torch.unsqueeze(torch.tensor(labels),0))


class FSDTestDataset(Dataset):
    def __init__(self, path, grouping_variables=['label']):
        data=pd.read_csv(path)
        self.data = data
        self.path = os.path.dirname(path).replace('meta','audio_test')
        classes = data.apply(lambda x:'_'.join(x[grouping_variables]), axis=1).to_frame(name='factor')
        self.id2label = {k:v for k,v in enumerate(sorted(self.data['label'].unique()))}
        self.label2id = {v:k for k,v in self.id2label.items()}
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
            audio, sr = sf.read(os.path.join(self.path, item['fname']))
            inputs.append(audio)
            labels.append(self.label2id[item['label']])
        return (inputs, torch.unsqueeze(torch.tensor(labels),0))
'''
