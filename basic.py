import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from COVIDDataset import COVIDDataset


    

if __name__ == "__main__":
    val_prop = 0.2
    grouping_variables = ['Covid_status', 'Gender'] ##For Stratified Split and Sampling
    data_path = '/home/icalloway/Side Projects/COVIDClassification/Data/Track1_Train/metadata.csv'
    samples = 1000
    batch_size = 8

    track1 = COVIDDataset(data_path, grouping_variables)
    train, val = train_test_split(track1.classes, test_size=val_prop, stratify=track1.classes['factor'])

    ##Train DataLoader
    track1_train = Subset(track1,train['idx'])
    track1_train_weights = [1/(len(track1.counts.keys())*track1.counts[track1.classes.loc[a]['factor']]) for a in train['idx']]
    train_sampler = WeightedRandomSampler(
        weights = track1_train_weights,
        replacement=True,
        num_samples = samples)

    train_loader = DataLoader(
        dataset=track1_train,
        batch_size = batch_size,
        sampler = train_sampler,
        pin_memory = True
    )

    ##Validation DataLoader
    track1_val = Subset(track1,val['idx'])
    track1_val_weights = [1/(len(track1.counts.keys())*track1.counts[track1.classes.loc[a]['factor']]) for a in val['idx']]
    val_sampler = WeightedRandomSampler(
        weights = track1_val_weights,
        replacement=True,
        num_samples = samples)

    val_loader = DataLoader(
        dataset=track1_val,
        batch_size = batch_size,
        sampler = val_sampler,
        pin_memory = True
    )


    

