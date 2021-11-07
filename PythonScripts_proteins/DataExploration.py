import pandas as pd

#checking how the randrange function works
from random import randrange
print(randrange(2))

#meant to be read in the School/ProteinTransformers/Scripts file
DataFilePath = "../degree_tokenized.csv"
ProteinData = pd.read_csv(DataFilePath, names=['Sequence','Degree','Tokenized Sequence'], skiprows=1)
print(len(list(ProteinData['Degree']))==len(list(ProteinData['Sequence'])))
print(ProteinData['Sequence'])
print(ProteinData['Degree'])

#imports
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import os
import pandas as pd
from pandas import DataFrame
import requests
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
from random import randrange
from sklearn.utils import shuffle

model_name = 'Rostlab/prot_bert_bfd'

class ProteinDegreeDataset(Dataset):

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd', max_length=1024):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        #for this dataset, the number of rows was determined to be 16215
        self.datasetLength = 16215

        #split the dataset into test, train, and validation
        self.trainSplit = .90
        self.testSplit = .5
        self.validSplit = .5

        self.datasetFilePath = "../degree_tokenized.csv"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

        self.max_length = max_length

        self.seqs = self.load_datasets(self.datasetFilePath)[split][0]
        self.labels = self.load_datasets(self.datasetFilePath)[split][1]

    def load_datasets(self,path):
        df = pd.read_csv(path,names=['Sequence','Degree','Tokenized Sequence'],skiprows=1)

        df = shuffle(df)

        df['labels'] = df['Degree']

        firstchunk = round(self.datasetLength * self.trainSplit)

        trainDf = df.iloc[0 : firstchunk]
        testDf = df.iloc[firstchunk : firstchunk + round(self.datasetLength * self.testSplit)]
        validDf = df.iloc[firstchunk + round(self.datasetLength * self.testSplit):]
        
        seq = list(df['Sequence'])
        label = list(df['labels'])

        datasets = {'train' : [list(trainDf['Sequence']), list(trainDf['labels'])], 'test' : [list(testDf['Sequence']), list(testDf['labels'])], 'valid' : [list(validDf['Sequence']), list(validDf['labels'])]}

        assert len(seq) == len(label)
        return datasets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        #this line looks to be specific to the dataset?
        #seq = re.sub(r"[UZOB]", "X", seq)

        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample

train_dataset = ProteinDegreeDataset(split="train", tokenizer_name=model_name, max_length=1024)
val_dataset = ProteinDegreeDataset(split="valid", tokenizer_name=model_name, max_length=1024)
test_dataset = ProteinDegreeDataset(split="test", tokenizer_name=model_name, max_length=1024)
