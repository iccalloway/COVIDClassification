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

from COVIDDataset import COVIDTest
from COVIDModels import COVIDWav2Vec, DenseNet

def test(model, out_path, data_loader):
    model.eval()
    
    with open(out_path, 'w') as f:
        with torch.no_grad():
            for i, (inputs, file, label) in enumerate(data_loader):
                if i == len(data_loader)-1: #last batch not same size as others
                    break
                out = model(inputs)
                prediction = torch.round(sigmoid(out))
                f.write("{} {}\n".format(file,round(prediction.item())))
    return
            

if __name__ == "__main__":
    device = torch.device("cuda")

    val_prop = 0.2
    grouping_variables = ['Covid_status', 'Gender'] ##For Stratified Split and Sampling
    data_path = './Data/Track1_Eval/test_metadata.csv' ##'./Data/Track1_Train/metadata.csv'
    out_path = "test_scores.txt" ##"val_scores.txt"
    samples = 1000
    batch_size = 1

    track1 = COVIDTest(data_path, [])

    test_loader = DataLoader(
        dataset=track1,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=track1.collate_batch
    )

    #model = DenseNet().to(device)
    model = COVIDWav2Vec(device).to(device)
    model.load_state_dict(torch.load('model.pt'))


    loss_fn = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()


    test(model, out_path, test_loader)
                
