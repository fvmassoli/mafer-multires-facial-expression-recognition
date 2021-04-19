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

from .base import FoldedBaseDataset


class OuluCasia(FoldedBaseDataset):
    """Class to handle OuluCasia data.
    The OuluCasia dataset is used ONLY to TEST the models
    
    """
    def __init__(self, dataset_folder: str, output_folder: str, transforms: torchvision.transforms = None, show_stats: bool = False):
        """Init the dataset class.

        Parameters
        ----------
        dataset_folder : str
            Path to folder cotaining the dataset
        output_folder : str
            Path to output folder where to save the database 
        transforms : torchvision.transforms
            Data augmentation transforms
        show_stats : bool
            Whether to show dataset stats or not
        
        """
        super(OuluCasia, self).__init__()

        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.transforms = transforms
        self.show_stats = show_stats

        self.idx_to_expression = {
                            0: 'Surprise',
                            1: 'Happiness',
                            2: 'Sadness',
                            3: 'Anger',
                            4: 'Fear',
                            5: 'Disgust'
                        }

        self.expression_to_idx = {
                            'Surprise': 0,
                            'Happiness': 1,
                            'Sadness': 2,
                            'Anger': 3,
                            'Fear': 4,
                            'Disgust': 5
                        }
        
        # Create the database if it does not exists otherwise load it from .csv file
        db_fname = join(self.output_folder, 'oulucasia_database.csv')
        print(db_fname)
        if not exists(db_fname):
            if not exists(self.output_folder):
                os.makedirs(self.output_folder)
            self.logger.info('Database not found... creating it')
            self.db = self.__create_database(db_fname=db_fname)
        else:
            self.db = pd.read_csv(db_fname)
        
        # Print data statistics
        if self.show_stats:
            self.__dataset_statistics()
        
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
        df = pd.DataFrame()

        imgs = []
        exprs = []
        for expr in listdir(self.dataset_folder):
            idx_for_expr = self.expression_to_idx[expr]
            for img in listdir(join(self.dataset_folder, expr)):
                imgs.append(join(self.dataset_folder, expr, img))
                exprs.append(idx_for_expr)

        # Create the database
        df = pd.DataFrame(
                    data=dict(
                            image_path=imgs, 
                            expression=exprs
                        ),
                    index=None
                )

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
        stats_table.field_names = ['Images', 'Number of expressions']
        stats_table.add_row([len(self.db), len(self.db.expression.unique())])
        print(stats_table)
        
        # Print expression statistics
        stats_table = PrettyTable()
        expressions = self.db.expression.unique()
        expressions.sort()
        stats_table.field_names = [self.idx_to_expression[label] for label in expressions]
        stats_table.add_row([len(self.db[self.db['expression']==expr]) for expr in expressions])
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
        
        emotion = self.db.iloc[idx].expression
        
        return img, emotion