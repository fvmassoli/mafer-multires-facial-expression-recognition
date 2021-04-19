import os
import cv2
import argparse
import prettytable
import numpy as np
from PIL import Image
from tqdm import tqdm
from os.path import join, exists

import sklearn.metrics as skm

from models import ModelLoader
from datasets.fer2013 import FER2013
from datasets.rafdb import RAFdb
from datasets.oulucasia import OuluCasia

import albumentations as A
from albumentations.pytorch import ToTensor

import torch
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import Dataset, DataLoader

import torchvision


def init_seeds(seed: int) -> str:
    """Set the seeds for reproducibility purposes.
    
    Parameters
    ----------
    seed : int
        Seed

    Returns
    -------
    device : str
        Device to use

    """
    if seed != -1:
        # cudnn.benchmark = True
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def create_experiment_folder(args: argparse.Namespace, dataset_name: str) -> str:
    """Create the current output experiment folder.

    Parameters
    ----------
        args : 
            Argument given by the user
        dataset_name : str
            Name of the dataset

    Returns
    ------
    exp_folder : str
        Path to current experiment output folder

    """
    output_main_folder = args.output_folder_path
    
    if not exists(output_main_folder):
        os.makedirs(output_main_folder)

    if dataset_name == 'affwild2':
        tmp = '-'.join([str(args.multi_res_training), args.optimizer, str(args.lr), str(args.weight_decay), str(args.momentum), str(args.nesterov), args.train_mode, dataset_name, str(args.batch_accumulation), str(args.batch_size)])
    else:
        tmp = '-'.join(['new', args.optimizer, str(args.lr), str(args.weight_decay), str(args.momentum), str(args.nesterov), dataset_name, str(args.batch_accumulation), str(args.batch_size)])

    exp_folder = join(output_main_folder, tmp)

    if not exists(exp_folder):
        os.makedirs(exp_folder)

    return output_main_folder, exp_folder


def create_output_test_folder(output_folder_path, dataset_name):
    """Create the test output folder.

    Parameters
    ----------
        output_folder_path : str
            Where to create the folder
        dataset_name : str
            Name of the dataset

    Returns
    ------
    exp_folder : str
        Path to current experiment output folder

    """
    outf = join(output_folder_path, dataset_name)
    if not exists(outf):
        os.makedirs(outf)
    return outf 


def get_model(model_base_path: str, num_classes: int, model_checkpoint: str, training_config: str, load_last_layer: bool) -> torch.nn.Module:
    """Init the model, load the checkpoint and freeze the parameters.
    
    Parameters
    ----------
    model_base_path : str
        Path to base model checkpoint
    num_classes : int
        Number of expressions in the dataset
    model_checkpoint : str
        Path to actual model checkpoint
    training_config : str
        Set the training to finetuning or transfer learning
    load_last_layer : bool
            Boolean used to decide wether or not load the last layer weights. 
    
    Returns
    -------
    model : torch.nn.Module
        Loaded model ready for training

    """
    print(f'\nLoading model: {model_checkpoint}')
    model_loader = ModelLoader(
                            model_base_path=model_base_path, 
                            num_classes=num_classes, 
                        )
    # Load model checkoint
    if model_checkpoint is not None:
        model_loader.load_model_checkpoint(model_checkpoint=model_checkpoint, load_last_layer=load_last_layer)
    print('Model loaded!')
    # Freeze layers for transfer learning
    if training_config != 'finetune':
        model_loader.freeze_params()
        print('Parameters freezed!')
    return model_loader.get_model()


def init_optimizer(model: torch.nn.Module, optimizer_name: str, lr: float, lr_ft: float, momentum: float, nesterov: bool, weight_decay: float, amsgrad: bool, beta1: float, beta2: float, training_config: str) -> torch.optim:
    """Init the optimizer.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    optimizer_name : str
        Optimizer name
    lr : float
        Learning rate
    lr_ft : float
        Learning rate for first layers only
    momentum : float
        Optimizer momentum
    nesterov : bool
        Apply Nesterov
    weight_decay : float
        Weight decay
    amsgrad : bool
        Use AmsGrad with Adam
    beta1 : float
        Beta 1 parameters for Adam
    beta2 : float
        Beta 2 parameters for Adam
    training_config : str
        Set the training to finetuning or transfer learning
    
    Returns
    -------
    optimzier : torch.optim
        Optimizer

    """
    ## Init the optimizer
    optim_kwargs = {
                'weight_decay': weight_decay
            }
    if optimizer_name == 'sgd':
        opt_fn = SGD
        optim_kwargs.update({
            'momentum': momentum,
            'nesterov': nesterov
        })
    else:
        opt_fn = Adam if optimizer_name == 'adam' else AdamW
        optim_kwargs.update({
            'betas':(beta1, beta2),
            'amsgrad': amsgrad
        })
    # Set the learning rate depending on the training configuration
    if training_config == 'finetune':
        return opt_fn([
                    {'params': torch.nn.Sequential(*(list(model.children())[:-1])).parameters(), 'lr': lr_ft},
                    {'params': model.classifier_1.parameters()}
                ],  
                lr=lr, 
                **optim_kwargs
            )
    else: # For transfer learning just train the last layer
        return opt_fn(params=model.classifier_1.parameters(), lr=lr, **optim_kwargs)
    

def init_datasets(dataset_name: str, dataset_folder: str, exp_folder: str) -> [Dataset, Dataset, Dataset]:
    """Init the datasets.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    dataset_folder : str
        Path to dataset main folder
    exp_folder : str
        Path to folder to save dataset database

    Returns
    -------
    datasets : [Dataset, Dataset, Dataset]
        List containing the initialized datasets
        
    """
    kwargs = dict(
                dataset_folder=dataset_folder, 
                output_folder=exp_folder, 
                transforms=None # we manually set them later
            )
    if dataset_name == 'fer2013':
        train_dset = FER2013(mode='Training', **kwargs)
        valid_dset = FER2013(mode='PublicTest', **kwargs)
        test_dset  = FER2013(mode='PrivateTest', **kwargs)
    elif dataset_name == 'rafdb':
        train_dset = RAFdb(mode=1, **kwargs)
        valid_dset = RAFdb(mode=-1, **kwargs)
        test_dset  = RAFdb(mode=0, **kwargs)
    else: # dataset_name == 'oulucasia'
        # train, valid and test set are the same since we need two differnt
        # transformations for k-folds
        dset_fn = OuluCasia
        train_dset = dset_fn(**kwargs)
        valid_dset = dset_fn(**kwargs)
        test_dset  = dset_fn(**kwargs)
        
    return train_dset, valid_dset, test_dset


def init_loader(dset: Dataset, batch_size: int, shuffle: bool = False) -> torch.utils.data.DataLoader:
    """Init the dataloader for the given dataset.
    
    Parameters
    ----------
    dset : Dataset
        The dataset
    batch_size : int
        Size of the batch
    shuffle : 
        True to shuffle the data
    
    Returns
    -------
    loader : torch.utils.data.DataLoader

    """
    return DataLoader(
                    dataset=dset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=8,
                    pin_memory=True
                )


def get_transforms(train: bool, dataset_name: str, data_mean: np.array) -> A.augmentations.transforms:
    """Get the transformations for data augmentation.
    
    Parameters
    ----------
    train : bool
        If augmentation is for training or not
    dataset_name : str
        Name of the dataset
    data_mean : np.array
        Mean for each channel

    Returns 
    -------
    Transforms : albumentations
        Traning transforms

    """
    def subtract_mean(x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Subtract the mean for each channel.
        
        Parameters
        ----------
        x : torch.Tensor
            The input image tensor
            
        Returns
        -------
        x : torch.Tensor
            Nomralized tensor image

        """
        x *= 255.
        if x.shape[0] == 1:  
            x = x.repeat(3, 1, 1) # FER2013 dataset returns b/w images
        x[0] -= data_mean[0]
        x[1] -= data_mean[1]
        x[2] -= data_mean[2]
        return x
    if train:
        if dataset_name == 'fer2013':
            return A.Compose([
                        A.RandomCrop(height=42, width=42, always_apply=True),
                        A.Resize(height=224, width=224, always_apply=True),
                        A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
                        A.Cutout(num_holes=10, max_h_size=30, max_w_size=30, fill_value=0, always_apply=True),
                        A.HorizontalFlip(p=0.5),
                        ToTensor(),
                        A.Lambda(name='subtract_mean', image=subtract_mean, always_apply=True, p=1)
                ])
        elif dataset_name == 'rafdb':
            return A.Compose([
                        A.Resize(256, 256, always_apply=True),
                        A.RandomCrop(224, 224, always_apply=True),
                        A.ToGray(p=0.2),
                        A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
                        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
                        A.Cutout(num_holes=10, max_h_size=20, max_w_size=20, fill_value=0, p=1),
                        A.HorizontalFlip(p=0.5),
                        ToTensor(),
                        A.Lambda(name='subtract_mean', image=subtract_mean, always_apply=True, p=1)
                    ])
        elif dataset_name == 'oulucasia':
            return A.Compose([
                        A.Resize(height=224, width=224, always_apply=True),
                        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.3, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                        A.Cutout(num_holes=10, max_h_size=10, max_w_size=10, fill_value=0, always_apply=True),
                        A.HorizontalFlip(p=0.5),
                        ToTensor(),
                        A.Lambda(name='subtract_mean', image=subtract_mean, always_apply=True, p=1)
                ])
        else: 
            return None
    else:
        if dataset_name == 'oulucasia':
            return A.Compose([
                            A.Resize(height=224, width=224, always_apply=True),
                            ToTensor(),
                            A.Lambda(name='subtract_mean', image=subtract_mean, always_apply=True, p=1)
                    ])
        elif dataset_name == 'fer2013' or dataset_name == 'rafdb':
            return A.Compose([
                        A.Resize(256, 256, always_apply=True),
                        A.CenterCrop(224, 224, always_apply=True),
                        ToTensor(),
                        A.Lambda(name='subtract_mean', image=subtract_mean, always_apply=True, p=1)
                    ])
        else: 
            return None


def eval_metrics(labels: torch.tensor, predictions: torch.tensor) -> [float, float, skm]:
    """Eval training metrics.
    
    Parameters
    ----------
    labels : torch.tensor
        Ground truth predictions
    predictions : torch.tensor
        Model predictions

    Returns
    ------
    accuracy : float
        Accuracy of the model
    f1_score : float 
        F1 score of the model
    confusion_matrix : sklearn.metrics
        Confusion matrix among all 7 expression classes

    """
    # Eval F1 score (macro-averaged)
    f1_score = skm.f1_score(labels, predictions, average='macro', zero_division=1)
    
    # Eval F1 score for each class
    classes_score = skm.f1_score(labels, predictions, average=None, zero_division=1)
    print(f'F1 score classes: {classes_score}')

    # Eval accuracy
    accuracy = skm.accuracy_score(labels, predictions)
    
    # Eval the confusion matrix
    confusion_matrix = skm.confusion_matrix(labels, predictions)

    return accuracy, f1_score, confusion_matrix

def class_accuracy(model: torch.nn.Module, loader: DataLoader, device: str) -> None :
    """Calculate the average of the accuracy of each class. Used to evaluate class accuracy on RAF-DB test set
    """
    n=0
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader, total=len(loader), desc='Model test', leave=False), 1):
            n += x.shape[0]
            
            x, y = x.to(device), y.to(device)

            _, outputs = model(x)
            _, predictions_ = torch.max(outputs, 1)

            c = (predictions_ == y).squeeze()
            for i in range(x.size()[0]):
                label = y[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(7):
        print(class_correct[i] / class_total[i])
    return


def eval_score_for_competition(f1_score: float, accuracy: float) -> float:
    """Evaluate the statistics required from the AffWild2 competition.

    Parameters
    ----------
    f1_score : float
        F1 score of the classifier
    accuracy : float
        Accuracy of the classifier 

    Returns
    ------
    stat : float
        Competition statistics

    """
    return (0.33*accuracy) + (0.67*f1_score)
