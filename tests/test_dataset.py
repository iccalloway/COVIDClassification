import unittest

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler

from COVIDDataset import *

class TestCOVIDDataset(unittest.TestCase):
    """
    Instantiate each dataset and run it through a training loop to test the
    collation function.
    """
    def setUp(self):
        self.data_base_dir = '/home/stic/Documents/dicova/data'
    
    def test_track1(self):
        val_prop = 0.2
        grouping_variables = ['Covid_status', 'Gender'] ##For Stratified Split and Sampling
        data_path = f'{self.data_base_dir}/DiCOVA_Train_Val_Data_Release/metadata.csv'
        samples = 1000
        batch_size = 2

        dataset = COVIDDataset(data_path, grouping_variables)
        train, val = train_test_split(dataset.classes, test_size=val_prop,
                stratify=dataset.classes['factor'])

        dataset_train = Subset(dataset, train['idx'])
        dataset_train_weights = [1/(len(dataset.counts.keys())*dataset.counts[dataset.classes.loc[a]['factor']]) if a in train['idx'] else 0 for a in range(len(dataset))]
        train_sampler = WeightedRandomSampler(
            weights = dataset_train_weights,
            replacement=True,
            num_samples = samples)

        train_loader = DataLoader(
            dataset=dataset_train,
            batch_size = batch_size,
            sampler = train_sampler, # this is necessar for some reason
            pin_memory = True,
            collate_fn = dataset.collate_batch
        )

        for i, (inputs, labels) in enumerate(train_loader):
            print(inputs)
            print(labels)
            break


    def test_track2(self):
        val_prop = 0.2
        grouping_variables = ['Covid_status', 'Gender'] ##For Stratified Split and Sampling
        data_path = f'{self.data_base_dir}/DiCOVA_Train_Val_Data_Release/metadata.csv'
        samples = 1000
        batch_size = 2

        data_path = f'{self.data_base_dir}/DiCOVA_Track_2_Release/metadata.csv'
        subdatasets = ['breathing-deep', 'counting-normal', 'vowel-e']

        for subdataset in subdatasets:
            dataset = DiCOVATrack2(data_path, grouping_variables, subdataset)
            train, val = train_test_split(dataset.classes, test_size=val_prop,
                    stratify=dataset.classes['factor'])

            dataset_train = Subset(dataset, train['idx'])
            dataset_train_weights = [1/(len(dataset.counts.keys())*dataset.counts[dataset.classes.loc[a]['factor']]) if a in train['idx'] else 0 for a in range(len(dataset))]
            train_sampler = WeightedRandomSampler(
                weights = dataset_train_weights,
                replacement=True,
                num_samples = samples)

            train_loader = DataLoader(
                dataset=dataset_train,
                batch_size = batch_size,
                sampler = train_sampler, # this is necessar for some reason
                pin_memory = True,
                collate_fn = dataset.collate_batch
            )

            for i, (inputs, labels) in enumerate(train_loader):
                print(inputs)
                print(labels)
                break

    def test_fsd_train(self):
        val_prop = 0.2
        # grouping_variables = ['label', 'manually_verified'] ##For Stratified Split and Sampling
        data_path = f'{self.data_base_dir}/Freesound/FSDKaggle2018.meta/train_post_competition.csv'
        dataset = FSDTrainDataset(data_path)
        samples = 1000
        batch_size = 2

        train, val = train_test_split(dataset.classes, test_size=val_prop,
                stratify=dataset.classes['factor'])


        dataset_train = Subset(dataset, train['idx'])
        dataset_train_weights = [1/(len(dataset.counts.keys())*dataset.counts[dataset.classes.loc[a]['factor']]) if a in train['idx'] else 0 for a in range(len(dataset))]
        train_sampler = WeightedRandomSampler(
            weights = dataset_train_weights,
            replacement=True,
            num_samples = samples)

        train_loader = DataLoader(
            dataset=dataset_train,
            batch_size = batch_size,
            sampler = train_sampler, # this is necessar for some reason
            pin_memory = True,
            collate_fn = dataset.collate_batch
        )

        for i, (inputs, labels) in enumerate(train_loader):
            print(inputs)
            print(labels)
            break

    def test_fsd_test(self):
        val_prop = 0.2
        data_path = f'{self.data_base_dir}/Freesound/FSDKaggle2018.meta/test_post_competition_scoring_clips.csv'
        dataset = FSDTestDataset(data_path)
        samples = 1000
        batch_size = 2

        train, val = train_test_split(dataset.classes, test_size=val_prop,
                stratify=dataset.classes['factor'])


        dataset_train = Subset(dataset, train['idx'])
        dataset_train_weights = [1/(len(dataset.counts.keys())*dataset.counts[dataset.classes.loc[a]['factor']]) if a in train['idx'] else 0 for a in range(len(dataset))]
        train_sampler = WeightedRandomSampler(
            weights = dataset_train_weights,
            replacement=True,
            num_samples = samples)

        train_loader = DataLoader(
            dataset=dataset_train,
            batch_size = batch_size,
            sampler = train_sampler, # this is necessar for some reason
            pin_memory = True,
            collate_fn = dataset.collate_batch
        )

        for i, (inputs, labels) in enumerate(train_loader):
            print(inputs)
            print(labels)
            break
