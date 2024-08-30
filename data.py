# -----------------------------------------------------------------------------
# Functions to create dataset and dataloader
# -----------------------------------------------------------------------------
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from util import data_util


get_dataset_funcs = {
    'cifar100_images': data_util.load_cifar100_images,
    'imagenet100': data_util.load_imagenet100,
    'tiny-imagenet-200': data_util.load_tiny_imagenet_200,
    'caltech101': data_util.load_caltech101,
    'aircraft': data_util.load_aircraft,
    'scars': data_util.load_stanford_cars,
    'cub200': data_util.load_CUB_200,
}

class create_ccd_dataset(Dataset):
    '''
    Input: dataset class and splitted data index list
    Return: a new dataset class that consists only the splitted data considering CCD stage
            where stage 0 is labelled data and stage > 0 is unlabelled data
    '''
    def __init__(self, dataset, transform, stage) -> None:
        super(create_ccd_dataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.batch_labeled_or_not = 1 if stage == 0 else 0

    def __getitem__(self, index):
        batch_data = cv2.imread(self.dataset['paths'][index])
        batch_data = cv2.cvtColor(batch_data, cv2.COLOR_BGR2RGB)
        batch_data = Image.fromarray(batch_data) 
        batch_label = self.dataset['labels'][index]
        batch_unique_index = self.dataset['uq_idx'][index]

        if self.transform is not None:
            batch_data = self.transform(batch_data)

        return batch_data, batch_label, batch_unique_index, np.array([self.batch_labeled_or_not])
    
    def __len__(self):
        return self.dataset['len']


class create_ccd_test_dataset(Dataset):
    '''
    Input: dataset class and splitted data index list
    Return: a new dataset class that consists only the splitted data
    '''
    def __init__(self, unlabelled_dataset, labelled_dataset, transform) -> None:
        super(create_ccd_test_dataset, self).__init__()
        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.transform = transform

    def __getitem__(self, index):
        if index < self.labelled_dataset['len']:
            batch_data = cv2.imread(self.labelled_dataset['paths'][index])
            batch_data = cv2.cvtColor(batch_data, cv2.COLOR_BGR2RGB)
            batch_data = Image.fromarray(batch_data) 
            batch_label = self.labelled_dataset['labels'][index]
            batch_unique_index = self.labelled_dataset['uq_idx'][index]
            batch_labeled_or_not = 1
    
        else:
            index = index - self.labelled_dataset['len']
            batch_data = cv2.imread(self.unlabelled_dataset['paths'][index])
            batch_data = cv2.cvtColor(batch_data, cv2.COLOR_BGR2RGB)
            batch_data = Image.fromarray(batch_data) 
            batch_label = self.unlabelled_dataset['labels'][index]
            batch_unique_index = self.unlabelled_dataset['uq_idx'][index]
            batch_labeled_or_not = 0

        if self.transform is not None:
            batch_data = self.transform(batch_data)

        return batch_data, batch_label, batch_unique_index, np.array([batch_labeled_or_not])
    
    def __len__(self):
        if self.unlabelled_dataset == None:
            return self.labelled_dataset['len']
        else:
            return self.labelled_dataset['len'] + self.unlabelled_dataset['len']
        

def add_dataset_attribute(dataset):
    dataset, index, transform = dataset
    data_path_list = []
    label_list = []
    np.random.shuffle(index)

    for i in index:
        data_path_list.append(dataset[0][i])
        label_list.append(dataset[1][i])

    dataset = {
        'paths': data_path_list,
        'labels': label_list,
        'uq_idx': index,
        'len': len(index)
    }
    return dataset, transform, max(index)


def combined_dataset(dataset_list, use_gt, save_path, eval_on_train=False):
    data_path_list, label_list, index = None, None, None

    for idx, dataset in enumerate(dataset_list):
        data_path_list  = dataset['paths'] if data_path_list is None else data_path_list + dataset['paths']
    
        if idx > 0 and use_gt == False:
            # use predicted labels
            print("use predicted labels from SS-K-means algorithm")
            if eval_on_train:
                predicted_label_path = open(os.path.join(save_path, 'pred_labels', f'pred_labels_stage_{idx}_train.txt'), 'r')
            else:
                predicted_label_path = open(os.path.join(save_path, 'pred_labels', f'pred_labels_stage_{idx}.txt'), 'r')
            predicted_label_list = [int(x) for x in predicted_label_path.readlines()]
            label_list = label_list + predicted_label_list
        else:
            # use ground truth labels for dataset from stage 0 or when use_gt is True
            label_list = dataset['labels'] if label_list is None else label_list + dataset['labels']
        index = dataset['uq_idx'] if index is None else np.concatenate((index, dataset['uq_idx']))

    return {
        'paths': data_path_list,
        'labels': label_list,
        'uq_idx': index,
        'len': len(index)
    }
        

def create_dataloader(args, dataset_i, stage_i):
    '''
    for stage == -1, dataloader containes dataset_val
    for stage n > 0, dataloader contains dataset_train_i + rehearsal dataset
    for stage == 0, dataloader contains dataset_train_labelled
    '''
    contrast_dataset_i = None
    dataloader_dict, dataset_dict = {}, {}

    # create dataloader for evaluation
    if stage_i == -1:
        dataset_i, transform, _ = add_dataset_attribute(dataset_i)
        dataset_i = create_ccd_dataset(dataset_i, transform, stage=0) # set stage=0 to get labelled data
        dataloader = torch.utils.data.DataLoader(
            dataset_i,
            batch_size=args.val_batch_size,
            num_workers=args.val_workers,
            pin_memory=args.pin_mem,
            shuffle=False
        )       
        return dataloader

    # create dataloader for testing
    if stage_i == -2:
        dataset_i_list = []
        for dataset_i_j in dataset_i:
            dataset_i_j, transform, _ = add_dataset_attribute(dataset_i_j)
            dataset_i_list.append(dataset_i_j)

        unlabelled_val_dataset = None
        if len(dataset_i) > 1: 
            if args.eval_version == 'ccd' and args.train == False:
                unlabelled_val_dataset, labelled_val_dataset = dataset_i_list[-1], dataset_i_list[:len(dataset_i_list) - 1]
                dataset_i = combined_dataset(
                    labelled_val_dataset, 
                    args.use_gt_for_discovered_data, 
                    args.save_path, 
                    args.transductive_evaluation
                )

            elif args.eval_version == 'gcd' or args.train == True:
                unlabelled_val_dataset, dataset_i = dataset_i_list[-1], dataset_i_list[0]

            else: raise ValueError('Eval {} is not supported'.format(args.eval_version))

        else:
            dataset_i = dataset_i_list[0]

        dataset_i = create_ccd_test_dataset(unlabelled_val_dataset, dataset_i, transform)

        dataloader = torch.utils.data.DataLoader(
            dataset_i,
            batch_size=args.val_batch_size,
            num_workers=args.val_workers,
            shuffle=False
        )
        return dataloader

    # create dataloader for training
    elif stage_i >= 0:
        dataset_i, transform, max_index = add_dataset_attribute(dataset_i)
        if not args.use_strong_aug:
            contrast_transform = data_util.ContrastiveLearningViewGenerator(base_transform=transform, n_views=args.n_views)
        else:
            strong_transform = data_util.get_strong_transform(args)
            contrast_transform = data_util.StrongWeakView(strong_transform, transform)

        # if stage_i == 0, create dataloader for labelled data, while for stage_i > 0, create dataloader for unlabelled data
        dataset_i_ = create_ccd_dataset(dataset_i, transform, stage=stage_i)
        contrast_dataset_i = create_ccd_dataset(dataset_i, contrast_transform, stage=stage_i)

        if stage_i == 0:
            # during stage 0, create sampler to balance the class distribution
            sample_weights = [1 for i in range(len(dataset_i_))]
            sample_weights = torch.DoubleTensor(sample_weights)
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(dataset_i_))
        
        else:
            # during discovery stage we should not assume any class distribution, i.e., whether it is balanced or not
            sampler = None

        dataset_dict['default'] = dataset_i_
        dataset_dict['contrast'] = contrast_dataset_i
            
        for dataset in dataset_dict:
            if dataset_dict[dataset] != None:
                dataloader = torch.utils.data.DataLoader(
                    dataset_dict[dataset],
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=args.pin_mem,
                    drop_last=True, 
                    sampler=sampler if dataset != 'default' and dataset != 'eval' else None,
                )
                dataloader_dict[dataset] = dataloader

        return dataloader_dict


def build_CCD_dataset(args, split):
    '''
    Input: dataset configuration
    Return: datasets for experimetnal task
    
    Note: dataset_train index 0 contains labelled data, index [1, n_stage] contain mix data
    '''
    dataset_name = args.dataset
    number_of_stage = args.n_stage
    ccd_dataset = [None] * (number_of_stage)
    datasets = list()
    dataset_l = None

    # Get dataset
    try:
        dataset_train, dataset_val = get_dataset_funcs[dataset_name](split)
    except:
        raise ValueError('Dataset {} not found.'.format(dataset_name))
    
    if split == 'train':
        transform = data_util.build_transform('default', args)
        dataset = dataset_train
    
    elif split == 'val':
        transform = data_util.build_transform('test', args)
        return (dataset_val, [x for x in range(len(dataset_val[1]))], transform)

    elif split == 'test': # for inductive test dataset
        transform = data_util.build_transform('test', args)
        dataset = dataset_val
    else:
        raise ValueError('Split {} is not supported for CCD training.'.format(split))
    
    # Get dataset classes
    dataset_target = np.array(dataset[1])
    for i in range(args.classes):

        class_i = np.where(np.array(dataset_target == i))[0]
        np.random.shuffle(class_i)

        # If class index is in labelled dataset, split it according to args.ccd_split_ratio for every stage
        if i < args.labelled_data:
            class_i_l, class_i_u = np.split(class_i, [int(len(class_i) * args.ccd_split_ratio[0][0])])
            dataset_l = class_i_l if dataset_l is None else np.concatenate((dataset_l, class_i_l), axis=0)
            for stage_i in range(number_of_stage):
                class_i_u, class_i_u_ = np.split(class_i_u, [int(len(class_i_u) * args.ccd_split_ratio[0][stage_i + 1])])
                ccd_dataset[stage_i] = class_i_u if ccd_dataset[stage_i] is None else np.concatenate((ccd_dataset[stage_i], class_i_u), axis=0)
                class_i_u = class_i_u_

        else:
            stage_selector = (i % args.labelled_data) // ((args.classes - args.labelled_data) // args.n_stage) 
            class_i_u = class_i
            for stage_i in range(number_of_stage - stage_selector):
                class_i_u, class_i_u_ = np.split(class_i_u, [int(len(class_i_u) * args.ccd_split_ratio[stage_selector + 1][stage_i])])
                ccd_dataset[stage_i + stage_selector] = class_i_u if ccd_dataset[stage_i + stage_selector] is None else np.concatenate((ccd_dataset[stage_i + stage_selector], class_i_u), axis=0)
                class_i_u = class_i_u_

    # Create training dataset according to index list
    # First index corresponds to labelled training dataset
    dataset_l = (dataset, dataset_l, transform)
    datasets.append(dataset_l)
    
    # Second index to rest correspond to mix training dataset (Known and unknown data)
    for dataset_u_i in ccd_dataset:
        dataset_u_i = (dataset, dataset_u_i, transform)
        datasets.append(dataset_u_i)

    return datasets