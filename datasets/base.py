import os
import PIL
import cv2
import logging
import numpy as np
import pandas as pd
from PIL import Image
from os import listdir
from os.path import join, exists
from prettytable import PrettyTable

import torchvision

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base class for all datasets.

    """
    def __init__(self):
        """Init the base dataset class."""
        super(BaseDataset, self).__init__()
        
        self.db = None
        self.db_mode = None
        self.idx_to_expression = None
        self.loader = self.__get_loader
        self.logger = logging.getLogger()
    
    def set_usage(self) -> None:
        """Select only a specific type of images."""
        self.db = self.db[self.db[self.db_mode] == self.mode]

    def get_expression_from_idx(self, label: int) -> str:
        """Returns the expression corresponding to the given label.
        
        Parameters
        ----------
        label : int
            Label of the required expression

        Returns
        ------
        emotion : str 
            Name of the expression that correspond to the given label 

        """
        return self.idx_to_expression[label]
    
    def get_training_classes_weights(self) -> np.array:
        """
        
        Returns
        ------
        weights : np.array
            Numpy array contatining the normalized weight for each training class

        """
        expressions = self.db.expression.unique() # Change self.db to self.curr_db for affwild2 dataset
        expressions.sort()
        weigths = np.asarray([len(self.db[self.db['expression']==expr]) for expr in expressions])
        normed_weights = np.asarray([1 - (w / sum(weigths)) for w in weigths])
        
        # Print weights for each class
        stats_table = PrettyTable()
        stats_table.title = 'Weights for each expression (Training)'
        stats_table.field_names = [self.idx_to_expression[int(label)] for label in expressions]
        stats_table.add_row([nw for nw in normed_weights])
        print(stats_table)
        
        return normed_weights

    def set_transforms(self, transforms: torchvision.transforms) -> None:
        """Set the transformation fot the dataset.
        
        Parameters
        ----------
        transforms : torchvision.transforms
            Data augmentation transforms
            
        """
        self.transforms = transforms

    @staticmethod
    def __get_loader(path: str) -> PIL.Image:
        """Set the image loader... 
        WARNING: the images are loaded as BGR due to the use of cv2.imread()
        
        Returns
        ------
        image : PIL.Image
            Image loaded from array

        """
        return cv2.imread(path)

    def __len__(self) -> int:
        """Returns the length of the database in the current mode.
        
        Returns
        ------
        int : int
            The number of samples in the current mode

        """
        return len(self.db)


class FoldedBaseDataset(BaseDataset):
    """Base class for folded-based datasets.

    """
    def __init__(self):
        """Init the base dataset class."""
        super(FoldedBaseDataset, self).__init__()
        self.fold_indices = None
