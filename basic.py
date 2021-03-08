import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import AdamW

from COVIDDataset import COVIDDataset
from COVIDModels import COVIDWav2Vec

if __name__ == "__main__":
    device = torch.device("cuda")

    val_prop = 0.2
    grouping_variables = ["Covid_status", "Gender"]  ##For Stratified Split and Sampling
    data_path = "/home/icalloway/Side Projects/COVIDClassification/Data/Track1_Train/metadata.csv"
    samples = 1000
    batch_size = 1
    epochs = 5
    gradient_accumulation = 500

    track1 = COVIDDataset(data_path, grouping_variables)
    train, val = train_test_split(
        track1.classes, test_size=val_prop, stratify=track1.classes["factor"]
    )

    ##Train DataLoader
    track1_train = Subset(track1, train["idx"])
    track1_train_weights = [
        1 / (len(track1.counts.keys()) * track1.counts[track1.classes.loc[a]["factor"]])
        if a in train["idx"]
        else 0
        for a in range(len(track1))
    ]
    train_sampler = WeightedRandomSampler(
        weights=track1_train_weights, replacement=True, num_samples=samples
    )

    train_loader = DataLoader(
        dataset=track1_train,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=track1.collate_batch,
    )

    ##Validation DataLoader
    track1_val = Subset(track1, val["idx"])
    track1_val_weights = [
        1 / (len(track1.counts.keys()) * track1.counts[track1.classes.loc[a]["factor"]])
        if a in val["idx"]
        else 0
        for a in range(len(track1))
    ]
    val_sampler = WeightedRandomSampler(
        weights=track1_val_weights, replacement=True, num_samples=samples
    )

    val_loader = DataLoader(
        dataset=track1_val, batch_size=batch_size, sampler=val_sampler, pin_memory=True
    )

    model = COVIDWav2Vec(device).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()

    ##Optimization
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ]
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
    optimizer.zero_grad()

    for a in range(epochs):
        print("Starting Epoch {}...\n=============".format(a))
        temp_loss = []
        accuracy = []
        for i, (inputs, labels) in enumerate(train_loader):
            try:
                out = model(inputs)
            except RuntimeError:
                torch.cuda.empty_cache()
                gc.collect()
                continue
            label = labels.to(device)
            loss = loss_fn(out.type(torch.float), label.type(torch.float))
            try:
                loss.backward()
            except RuntimeError:
                torch.cuda.empty_cache()
                gc.collect()
                continue
            temp_loss.append(loss.item())
            accuracy.append(round(label.item()) == round(sigmoid(out).item()))
            # print(label, sigmoid(out), accuracy[-1])
            if (i + 1) % gradient_accumulation == 0:
                if len(temp_loss) > 0:
                    mean_loss = sum(temp_loss) / len(temp_loss)
                    print("Training Loss: {}".format(mean_loss))
                    temp_loss = []
                if len(accuracy) > 0:
                    mean_metric = sum(accuracy) / len(accuracy)
                    print("Training Accuracy: {}".format(mean_metric))
                    accuracy = []
                print("\n")
                optimizer.step()
                optimizer.zero_grad()
