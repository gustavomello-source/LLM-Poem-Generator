import json
import os
from torch.utils.data import Dataset

class PoemDataset(Dataset):
    '''
    Class that represents the dataset of poems and implements the Dataset class from PyTorch.
    This class is responsible for loading the poems from a json file, processing them and 
    saving them to a text file.

    Attributes:
        poems (str): The path of the json file containing the poems.
    
    Methods:
        __init__(self, poems): Constructor of the class.
        __len__(self): Returns the number of poems in the dataset.
        __getitem__(self, idx): Returns a poem from the dataset.
        poems(self): Getter of the poems attribute.
        poems(self, value): Setter of the poems attribute.
    '''
    def __init__(self, poems, tokenizer, max_length=512):
        '''
        Constructor of the PoemDataset class.

        Args:
            poems (str): The path of the json file containing the poems.
            tokenizer (transformers.AutoTokenizer): The tokenizer for the model.
            max_length (int): The maximum length of the input sequence.

        Raises:
            ValueError: If the poems is None or an empty string.
        '''
        self.encodings = tokenizer(
            poems, 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            return_tensors="pt"
        )

    def __len__(self):
        '''
        Method returns the number of poems in the dataset.

        Returns:
            int: The number of poems in the dataset.
        '''
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        '''
        Method returns a poem from the dataset.

        Args:
            idx (int): The index of the poem to return.

        Returns:
            dict: The input_ids and attention_mask of the poem.
        '''
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx]
        }
    
    @property
    def poems(self):
        '''
        Method returns the poems attribute.

        Returns:
            str: The path of the text file containing the poems.
        '''
        return self.__poems
    
    @poems.setter
    def poems(self, value):
        '''
        Method receives the path of the text file containing the poems, validates and process it.

        Args:
            value (str): The path of the text file containing the poems.
        Raises:
            ValueError: If the value is None or an empty string.
        '''
        if value is None or value == '':
            raise ValueError("Poems cannot be None or an empty string.")

        self.__poems = value