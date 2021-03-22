from torch.utils.data import Dataset#*
import lmdb
import torchvision
import pandas as pd
import numpy as np
import pickle
import torch
import librosa
import torchaudio
import random
from PIL import Image
from audio import gain_db_to_ratio, max_dbfs, normalize_audio, AUDIO_TYPE_NP, AUDIO_TYPE_PCM, AUDIO_TYPE_OPUS
from helpers import LimitingPool, int_range, float_range, pick_value_from_range, tf_pick_value_from_range, MEGABYTE
import tensorflow as tf

class MelSpectrogram(object):
	def __init__(self, bins, mode):
		self.window_length = [25, 50, 100]
		self.hop_length = [10, 25, 50]
		self.fft = 22050
		self.melbins = bins
		self.mode = mode
		self.sr = 22050
		self.length = 250
	def __call__(self, value):
		sample = value
		limits = ((-2, 2), (0.9, 1.2))

		if self.mode=="train":
			pitch_shift = np.random.randint(limits[0][1], limits[0][1] + 1)
			time_stretch = np.random.random() * (limits[1][1] - limits[1][0]) + limits[1][0]
			new_audio = librosa.effects.time_stretch(librosa.effects.pitch_shift(sample, self.sr, pitch_shift), time_stretch)
		else:
			pitch_shift = 0
			time_stretch = 1
			new_audio = sample
		specs = []
		for i in range(len(self.window_length)):
			clip = torch.Tensor(new_audio)

			window_length = int(round(self.window_length[i]*self.sr/1000))
			hop_length = int(round(self.hop_length[i]*self.sr/1000))
			spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_fft=self.fft, win_length=window_length, hop_length=hop_length, n_mels=self.melbins)(clip)
			eps = 1e-6
			spec = spec.numpy()
			spec = np.log(spec+ eps)
			spec = np.asarray(torchvision.transforms.Resize((128, self.length))(Image.fromarray(spec)))
			specs.append(spec)
		specs = np.array(specs).reshape(-1, 128, self.length)
		specs = torch.Tensor(specs)
		return specs

class AudioDataset(Dataset):
	def __init__(self, data, transforms=None):
		self.transforms = transforms
		self.data = []
		self.length = 250
		self.data = data
	def __len__(self):
		if self.transforms.mode == "train":
			return 2*len(self.data)
		else:
			return len(self.data)
	def __getitem__(self, idx):
		if idx >= len(self.data):
			new_idx = idx - len(self.data)
			entry = self.data[new_idx]
			if self.transforms:
				values = self.transforms(entry["audio"])
		else:
			entry = self.data[idx]
			values = torch.Tensor(entry["values"].reshape(-1, 128, self.length))
		target = torch.LongTensor([entry["target"]])
		return (values, target)

def fetch_dataloader(batch_size, num_workers, mode):
	transforms = MelSpectrogram(128, mode)
	dataset = AudioDataset(transforms=transforms)
	dataloader = DataLoader(dataset,shuffle=True, batch_size=batch_size, num_workers=num_workers)
	return dataloader
	
class TimeMask():
	def __init__(self):
    		pass
	def __call__(self, data):
		return torchaudio.transforms.TimeMasking(0.1).forward(data)

class Volume():
	def __init__(self, gain=0.3):
		self.gain = gain
	def __call__(self, data):
		return torchaudio.transforms.Vol(self.gain).forward(data)

class FrequencyMask():
	def __init__(self, mask_param = 3):
		self.mask_param = mask_param
	def __call__(self, data):
		return torchaudio.transforms.FrequencyMasking(self.mask_param).forward(data)