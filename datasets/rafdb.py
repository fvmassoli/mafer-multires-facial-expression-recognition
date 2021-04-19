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


class RAFdb(BaseDataset):
    """Class to handle RAFdb data.
    The RAFdb dataset is used ONLY to TEST the models
    
    """
    def __init__(self, dataset_folder: str, output_folder: str, mode: str, transforms: torchvision.transforms = None, show_stats: bool = False):
        """Init the dataset class.

        Parameters
        ----------
        dataset_folder : str
            Path to folder cotaining the dataset
        output_folder : str
            Path to output folder where to save the database 
        mode : str
            Set the specific data split, i.e., train, valid or test
        transforms : torchvision.transforms
            Data augmentation transforms
        show_stats : bool
            Whether to show dataset stats or not
        
        """
        super(RAFdb, self).__init__()
        
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.transforms = transforms
        self.show_stats = show_stats
        
        self.mode = mode
        self.db_mode = 'train'

        self.transforms = transforms

        self.idx_to_expression = {
                            0: 'Surprise',
                            1: 'Fear',
                            2: 'Disgust',
                            3: 'Happiness',
                            4: 'Sadness',
                            5: 'Anger',
                            6: 'Neutral'
                        }

        # Create the database if it does not exists otherwise load it from .csv file
        db_fname = join(self.output_folder, 'rafdb_database.csv')
        if not exists(db_fname):
            if not exists(self.output_folder):
                os.makedirs(self.output_folder)
            self.logger.info('Database not found... creating it')
            self.db = self.__create_database(db_fname=db_fname)
        else:
            self.db = pd.read_csv(db_fname)
        
        # Print data statistics
        if mode == 1 and self.show_stats:
            self.__dataset_statistics()

        # Set the proper split
        self.set_usage()

    def __create_database(self, db_fname: str) -> pd.DataFrame:
        """Create the database and save it in a .csv file.
        
        Parameters
        ----------
        db_fname : str
            Path to output database file

        Returns
        ------
        dataframe : pd.DataFrame
            The database containing dataset infos

        """
        # Read the emotion files containing the list of images with the corresponding emotion label
        with open(join(self.dataset_folder, 'EmoLabel/list_patition_label.txt'), 'r') as f:
            lines = list(map(str.rstrip, f.readlines()))
        
        # Create the database and save it to a .csv file
        df = pd.DataFrame(
                    data=dict(
                        image_path=list(map(lambda x : join(self.dataset_folder, 'Image/aligned', x.split(' ')[0].split('.')[0]+'_aligned.jpg'), lines)),
                        expression=list(map(lambda x : int(x.split(' ')[1])-1, lines)), # The '-1' is because the expressions labels start from 1 and this rasise CUDA-assertion error when evaluating the loss
                        train=[int('train' in path) for path in list(map(lambda x : x.split(' ')[0], lines))]
                    ),
                    index=None     
                )
        
        # Shuffle before select validation images
        df = df.sample(frac = 1)
        df = df.sample(frac = 1)

        # For each expression, take 5% of images for validation set
        expressions = df.expression.unique()
        for expr in expressions:
            nb_valid_imgs = int(len(df[df.expression==expr])*0.05)
            index = df[(df.expression==expr) & (df.train==1)].iloc[:nb_valid_imgs].index
            df.loc[index, 'train'] = -1
            
        df.to_csv(db_fname)
        self.logger.info(f'Database saved at: {db_fname}')

        return df
    
    def __dataset_statistics(self) -> None:
        """Use PrettyTable package to print dataset info.
        
        """
        # Print the expressions
        print(f'Available expression: {self.idx_to_expression.values()}')
        
        # Print dataset cardinalities
        stats_table = PrettyTable()
        stats_table.title = 'Overall Stats'
        stats_table.field_names = ['Images', 'Train Samples', 'Test Samples', 'Number of expressions']
        stats_table.add_row([len(self.db), len(self.db[self.db['train'] == 1]), len(self.db[self.db['train'] == 0]), len(self.db.expression.unique())])
        print(stats_table)
        
        # Print expression statistics
        stats_table = PrettyTable()
        expressions = self.db.expression.unique()
        expressions.sort()
        stats_table.field_names = [self.idx_to_expression[label] for label in expressions]
        stats_table.add_row([len(self.db[self.db['expression']==expr]) for expr in expressions])
        print(stats_table)

        def eval_class_cardinality(expressions: list, train: str) -> [list, list]:
            cardinality = [len(self.db[(self.db['expression']==expr) & (self.db['train']==train)]) for expr in expressions]
            frac = [len(self.db[(self.db['expression']==expr) & (self.db['train']==train)])/len(self.db[self.db['train']==train]) for expr in expressions]
            return cardinality, frac

        expressions = self.db.expression.unique()
        expressions.sort()
        
        # Print expressions statistics --> Training 
        stats_table = PrettyTable()
        stats_table.title = 'Training Samples'
        stats_table.field_names = [self.idx_to_expression[label] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train=1)
        stats_table.add_row(cardinality)
        stats_table.add_row(frac)
        print(stats_table)

        # Print expressions statistics --> Validation 
        stats_table = PrettyTable()
        stats_table.title = 'Validation Samples'
        stats_table.field_names = [self.idx_to_expression[label] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train=-1)
        stats_table.add_row(cardinality)
        stats_table.add_row(frac)
        print(stats_table)

        # Print expressions statistics --> Test 
        stats_table = PrettyTable()
        stats_table.title = 'Test Samples'
        stats_table.field_names = [self.idx_to_expression[label] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train=0)
        stats_table.add_row(cardinality)
        stats_table.add_row(frac)
        print(stats_table)

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
        img = self.loader(self.db.iloc[idx].image_path)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        expression = self.db.iloc[idx].expression
        
        return img, expression
