import gc
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path


from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import AdamW

from COVIDDataset import COVIDDataset
from COVIDModels import COVIDWav2Vec, DenseNet, COVIDFairseq

def save_model(model, path):
    model = model.module if hasattr(model, "module") else model
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))

def rename_state_dict_keys(source, key_transformation, target=None):
    from collections import OrderedDict
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source)['model']
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    return new_state_dict

def fix(old_key):
    import re
    return re.sub('^encoder', 'model.encoder', old_key)


def evaluate(model, data_loader):
    model.eval()
    
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for i, (inputs, label) in enumerate(data_loader):
            label = label.cuda().T
            try:
                out = model(inputs)
            except RuntimeError:
                print("skipping")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            #out = model(inputs)
            prediction = torch.round(sigmoid(out))
            
            val_preds.append(prediction)
            val_labels.append(label)
            torch.cuda.empty_cache()
    if len(val_preds) < 1:
        print("No Predictions")
        return 0, 0
    else:
        val_preds = torch.cat(val_preds).squeeze(-1)
        val_labels = torch.cat(val_labels).squeeze(-1)
        val_accuracy = torch.sum(val_preds == val_labels).item()/len(val_preds)
        fpr, tpr, thresholds = roc_curve(val_labels.tolist(), val_preds.tolist(), pos_label=1)
        print(auc)
        return val_accuracy, auc(fpr, tpr)

if __name__ == "__main__":
    device = torch.device("cuda")

    val_prop = 0.2
    grouping_variables = ['Covid_status', 'Gender'] ##For Stratified Split and Sampling
    data_path = '/home/icalloway/Side Projects/COVIDClassification/Data/Track1_Train/metadata.csv'
    samples = 1000
    batch_size = 1
    epochs = 20
    gradient_accumulation = 256

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
        weights = track1_train_weights,
        replacement=True,
        num_samples = len(train) #samples
    )

    train_loader = DataLoader(
        dataset=track1_train,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=track1.collate_batch,
        drop_last=True
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
        weights = track1_val_weights,
        replacement=True,
        num_samples = len(val) #samples
    )
    val_loader = DataLoader(
        dataset=track1_val,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        collate_fn = track1.collate_batch,
        drop_last=True
    )

    #model = DenseNet().to(device)
    #model = COVIDWav2Vec(device).to(device)
    model = COVIDFairseq(device=device, path='wav2vec-pretrained-checkpoints/best_track1_long.pt').to(device)


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
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    optimizer.zero_grad()

    best_acc = 0
    accuracy, val_auc = evaluate(model, val_loader)
    print("Baseline: Accuracy - {} AUC - {}".format(accuracy, val_auc))
    for a in range(epochs):
        print("Starting Epoch {}...\n=============".format(a))
        temp_loss = []
        accuracy = []
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            try:
                out = model(inputs)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                gc.collect()
                raise
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
            torch.cuda.empty_cache()
        accuracy, val_auc = evaluate(model, val_loader)
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, './')
        print("Epoch {}: Accuracy - {} AUC - {}".format(a,accuracy, val_auc))
