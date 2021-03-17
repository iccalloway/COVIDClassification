import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import AdamW

from COVIDDataset import COVIDDataset
from COVIDModels import COVIDWav2Vec, DenseNet

def evaluate(model, data_loader):
    model.eval()
    
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for i, (inputs, label) in enumerate(data_loader):
            if i == len(data_loader)-1: #last batch not same size as others
                break
            if model_type == "cnn":
                inputs = torch.stack([x.cuda() for x in inputs])
            label = label.cuda().T

            out = model(inputs)
            prediction = torch.round(sigmoid(out))
            
            val_preds.append(prediction)
            val_labels.append(label)
    val_preds = torch.cat(val_preds).squeeze(-1)
    val_labels = torch.cat(val_labels).squeeze(-1)
    val_accuracy = torch.sum(val_preds == val_labels).item()/len(val_preds)
    fpr, tpr, thresholds = roc_curve(val_labels.tolist(), val_preds.tolist(), pos_label=1)
    return val_accuracy, auc(fpr, tpr)

if __name__ == "__main__":
    device = torch.device('cuda')

    val_prop = 0.2
    grouping_variables = ['Covid_status', 'Gender'] ##For Stratified Split and Sampling
    data_path = '/home/izimmerman/Documents/covid_cough/DiCOVA_Train_Val_Data_Release/metadata.csv'
    samples = 1000
    batch_size = 8
    epochs = 5
    gradient_accumulation = 100//batch_size
    model_type = 'cnn'
    

    track1 = COVIDDataset(data_path, grouping_variables, cnn = True if model_type != 'wav2vec' else False)
    train, val = train_test_split(track1.classes, test_size=val_prop, stratify=track1.classes['factor'])

    ##Train DataLoader
    track1_train = Subset(track1,train['idx'])
    track1_train_weights = [1/(len(track1.counts.keys())*track1.counts[track1.classes.loc[a]['factor']]) if a in train['idx'] else 0 for a in range(len(track1))]
    train_sampler = WeightedRandomSampler(
        weights = track1_train_weights,
        replacement=True,
        num_samples = samples)

    train_loader = DataLoader(
        dataset=track1_train,
        batch_size = batch_size,
        sampler = train_sampler,
        pin_memory = True,
        collate_fn = track1.collate_batch
    )

    ##Validation DataLoader
    track1_val = Subset(track1,val['idx'])
    track1_val_weights = [1/(len(track1.counts.keys())*track1.counts[track1.classes.loc[a]['factor']]) if a in val['idx'] else 0 for a in range(len(track1))]
    val_sampler = WeightedRandomSampler(
        weights = track1_val_weights,
        replacement=True,
        num_samples = 500)#samples

    val_loader = DataLoader(
        dataset=track1_val,
        batch_size = batch_size,
        sampler = val_sampler,
        pin_memory = True,
        collate_fn = track1.collate_batch
    )
    if model_type == "cnn":
        model = DenseNet().to(device)
    else:
        model = COVIDWav2Vec(device).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()

    ##Optimization
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    optimizer.zero_grad()

    writer = SummaryWriter("summary/densenet_aug4", purge_step=0)

    for a in range(epochs):
        print("Starting Epoch {}...\n=============".format(a))
        temp_loss=[]
        accuracy = []
        for i, (inputs, labels) in enumerate(train_loader):
            step = a * len(train_loader) + i + 1
            if model_type != 'wav2vec':
                inputs = torch.stack([x.cuda() for x in inputs])
            # try:
            out = model(inputs)
            # except RuntimeError:
            #     torch.cuda.empty_cache()
            #     gc.collect()
            #     continue
            label = labels.to(device).T
            loss = loss_fn(out.type(torch.float),label.type(torch.float)) 
            try:
                loss.backward()
            except RuntimeError:
                torch.cuda.empty_cache()
                gc.collect()
                continue
            temp_loss.append(loss.item())
            #accuracy.append(round(label.item()) == round(sigmoid(out).item()))
            accuracy.append(torch.sum(torch.round(sigmoid(out)) == label).item()/batch_size)
            del out
            if (i+1) % gradient_accumulation == 0:
                if len(temp_loss) > 0:
                    mean_loss = sum(temp_loss)/len(temp_loss)
                    print("Training Loss: {}".format(mean_loss))
                    writer.add_scalar("train/loss", mean_loss, step)
                    temp_loss = []
                if len(accuracy) > 0:
                    mean_metric = sum(accuracy)/len(accuracy)
                    print("Training Accuracy: {}".format(mean_metric))
                    writer.add_scalar("train/accuracy", mean_metric, step)
                    accuracy = []
                    val_accuracy, val_auc = evaluate(model, val_loader)
                    print("Validation Accuracy: {}".format(val_accuracy))
                    print("Validation AUC: {}".format(val_auc))
                    writer.add_scalar("val/accuracy", val_accuracy, step)
                    writer.add_scalar("val/auc", val_auc, step)
                    print('\n')
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()        
                
