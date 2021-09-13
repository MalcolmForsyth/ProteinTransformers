#imports
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, BertConfig, BertForSequenceClassification, AutoModelForSequenceClassification, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from scipy.stats import spearmanr
import re

model_name = "Rostlab/prot_bert_bfd"

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
        self.testSplit = .05
        self.validSplit = .05

        self.datasetFilePath = "degree_tokenized.csv"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

        self.max_length = max_length

        self.seqs = self.load_datasets(self.datasetFilePath)[split][0]
        self.labels = self.load_datasets(self.datasetFilePath)[split][1]

    def load_datasets(self,path):
        df = pd.read_csv(path,names=['Sequence','Degree','Tokenized Sequence'],skiprows=1)

        df = shuffle(df)

        df['Degree'] = np.log(df['Degree'])
        df['Degree'] = (df['Degree'] - np.mean(df['Degree']) )/ np.std(df['Degree'])
        df['labels'] = df['Degree']
        firstchunk = round(self.datasetLength * self.trainSplit)

        trainDf = df.iloc[0 : firstchunk]
        testDf = df.iloc[firstchunk : firstchunk + round(self.datasetLength * self.testSplit)]
        validDf = df.iloc[firstchunk + round(self.datasetLength * self.testSplit):]

        seq = list(df['Sequence'])
        label = list(df['labels'].astype(float))

        

        datasets = {'train' : [list(trainDf['Sequence']), list(trainDf['labels'].astype(float))], 'test' : [list(testDf['Sequence']), list(testDf['labels'].astype(float))], 'valid' : [list(validDf['Sequence']), list(validDf['labels'].astype(float))]}

        assert len(seq) == len(label)
        return datasets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)

        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample

train_dataset = ProteinDegreeDataset(split="train", tokenizer_name=model_name, max_length=1024)
val_dataset = ProteinDegreeDataset(split="valid", tokenizer_name=model_name, max_length=1024)
test_dataset = ProteinDegreeDataset(split="test", tokenizer_name=model_name, max_length=1024)



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    meanSquareError = mean_squared_error(labels, preds)
    residuals = []
    #again, dataset size
    for i in range(len(labels)):
        residuals.append(labels[i] - preds[i])
    df = pd.DataFrame({'freq': residuals})
    #for when we may want the residual plot
    #df.groupby('freq', as_index=False).size().plot(kind='bar')
    #plt.show()
    residualStandardDeviation = np.std(residuals)
    spearmancoeffiecient = spearmanr(labels, preds)
    return {
        'meanSquareError' : meanSquareError,
        'residualStandardDeviation' : residualStandardDeviation,
        'spearmanr' : spearmancoeffiecient[0]
    }

print(train_dataset[1])
print(val_dataset[1])
print(test_dataset[2])
