import os
import PIL
import cv2
import h5py
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from os import listdir
from os.path import join, exists
from prettytable import PrettyTable, from_csv

import torch
import torchvision

from .base import BaseDataset


class FER2013(BaseDataset):
    """Class to handle FER2013 data.
    The FER2013 dataset is used ONLY to TEST the models
    
    """
    def __init__(self, dataset_folder: str, mode: str, output_folder: str, transforms: torchvision.transforms = None, show_stats: bool = False):
        """Init the dataset class.

        Parameters
        ----------
        dataset_folder : str
            Path to the dataset
        mode : str
            Set the specific data split, i.e., train, valid or test
        output_folder : str
            Currently not used
        transforms : torchvision.transforms
            Data augmentation transforms
        show_stats : bool
            Whether to show dataset stats or not
        
        """
        super(FER2013, self).__init__()

        self.transforms = transforms
        self.show_stats = show_stats
        self.mode = mode
        
        self.database_path = join(dataset_folder, 'icml_face_data.csv')
        self.db_mode = 'Usage'

        self.transforms = transforms
        self.db = pd.read_csv(self.database_path)
        self.db = self.db[self.db['emotion'] != 7] # Remove the contempt

        # Rename columns so to have always same across datasets
        self.db = self.db.rename(columns={'emotion': 'expression', self.db.columns[1]: 'Usage', self.db.columns[-1]: 'pixels'})

        self.idx_to_expression = {
                            0: 'Angry', 
                            1: 'Disgust', 
                            2: 'Fear', 
                            3: 'Happy', 
                            4: 'Sad', 
                            5: 'Surprise', 
                            6: 'Neutral'
                        }
        
        if mode == 'Training' and self.show_stats: # Print stats only once
            self.__dataset_statistics()

        # Select the specific data split -- This call must follow the one for __dataset_statistics() 
        self.set_usage()

    def __dataset_statistics(self) -> None:
        """Use PrettyTable package to print dataset info.
        
        """
        # Print the expressions
        print(f'Available expressions: {self.idx_to_expression.values()}')
        
        # Print dataset cardinalities
        stats_table = PrettyTable()
        stats_table.field_names = ['Train samples', 'PublicTest samples', 'PrivateTest samples', 'Number of expressions']
        stats_table.add_row([len(self.db[self.db['Usage'] == 'Training']), len(self.db[self.db['Usage'] == 'PublicTest']), len(self.db[self.db['Usage'] == 'PrivateTest']), len(self.db.expression.unique().tolist())])
        print(stats_table)
        
        def eval_class_cardinality(expressions: list, train: str) -> [list, list]:
            cardinality = [len(self.db[(self.db['expression']==expr) & (self.db['Usage']==train)]) for expr in expressions]
            frac = [len(self.db[(self.db['expression']==expr) & (self.db['Usage']==train)])/len(self.db[self.db['Usage']==train]) for expr in expressions]
            return cardinality, frac

        expressions = self.db.expression.unique()
        expressions.sort()
        
        # Print expressions statistics --> Training 
        stats_table = PrettyTable()
        stats_table.title = 'Training Samples'
        stats_table.field_names = [self.idx_to_expression[label] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train='Training')
        stats_table.add_row(cardinality)
        stats_table.add_row(frac)

        # Print expressions statistics --> Validation 
        stats_table2 = PrettyTable()
        stats_table2.title = 'Validation Samples'
        stats_table2.field_names = [self.idx_to_expression[label] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train='PublicTest')
        stats_table2.add_row(cardinality)
        stats_table2.add_row(frac)

        # Print expressions statistics --> Test 
        stats_table3 = PrettyTable()
        stats_table3.title = 'Test Samples'
        stats_table3.field_names = [self.idx_to_expression[label] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train='PrivateTest')
        stats_table3.add_row(cardinality)
        stats_table3.add_row(frac)
        
        print(stats_table)
        print(stats_table2)
        print(stats_table3)

    def __getitem__(self, idx: int) -> [torch.Tensor, int]:
        """Load data. 
        
        Parameters:
        -----------
        idx : int
            Index of the sample to load

        Rerturn
        -------
        image : torch.Tensor
            The loaded tensor
        expression : int
            The label correspoding to the face expression

        """
        # Convert string of pixles to list of integers and then wrap it with a torch.Tensor
        img = np.asarray(list(map(int, self.db.iloc[idx].pixels.split(' '))))
        
        # Reshape the list of pixels to a 48x48 image
        img = img.reshape(48, 48, 1).astype('uint8')
        
        #img = img.astype('uint8')

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        labels = self.db.iloc[idx].expression

        return img, labels
