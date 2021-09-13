#imports
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Trainer, TrainingArguments, BertConfig, BertForSequenceClassification, AutoModelForSequenceClassification, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.utils import shuffle
import re

class ProteinDegreeDataset(Dataset):

    def __init__(self, split="train", max_length=330):
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

        self.trainFilePath = './sorted_train.csv'
        self.testFilePath = './sorted_test.csv'
        self.validFilePath = './sorted_valid.csv'
        if split=="train":
            self.seqs, self.labels = self.load_dataset(self.trainFilePath)
        elif split=='test':
            self.seqs, self.labels = self.load_dataset(self.testFilePath)
        elif split=='valid':
            self.seqs, self.labels = self.load_dataset(self.validFilePath)

        self.tokenizer = RobertaTokenizerFast.from_pretrained("../models/proberta")

        self.max_length = max_length

    def load_dataset(self,path):
        df = pd.read_csv(path,names=['Labels', 'Sequence','Degree','Tokenized Sequence'],skiprows=1)

        df['labels'] = df['Labels']
        df['Degree'] = np.log(df['Degree'])
        df['Degree'] = (df['Degree'] - np.mean(df['Degree']) )/ np.std(df['Degree'])
    
        seq = list(df['Sequence'])
        label = list(df['labels'].astype(float))

        assert len(seq) == len(label)
        return seq, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)

        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        '''
        print("sample")
        print(sample)
        '''
        return sample

train_dataset = ProteinDegreeDataset(split="train", max_length=330)
val_dataset = ProteinDegreeDataset(split="valid", max_length=330)
test_dataset = ProteinDegreeDataset(split="test", max_length=330)
'''
print("train")
print(train_dataset)
print("validation")
print(val_dataset)
print("test")
print(test_dataset)
'''


def compute_metrics(pred):
    print("preds")
    print(pred)
    labels = pred.label_ids
    preds = pred.predictions
    '''
    print("labels")
    print(labels)
    print("predictions")
    print(preds)
    '''
    meanSquareError = mean_squared_error(labels, preds)
    residuals = []
    for i in range(len(labels)):
        residuals.append(labels[i] - preds[i])
    df = pd.DataFrame({'freq': residuals})
    residualStandardDeviation = np.std(residuals)
    spearmancoeffiecient = spearmanr(labels, preds)
    return {
        'meanSquareError' : meanSquareError,
        'residualStandardDeviation' : residualStandardDeviation,
        'spearmanr' : spearmancoeffiecient[0]
    }

def model_init():
    #config = BertConfig(num_labels=1, hidden_size=1024, num_attention_heads=16) 
    #model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    #return model
    model = RobertaForMaskedLM.from_pretrained('../models/proberta3/checkpoint-320000')
    return model

stop_callback = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0 
) 
training_args = TrainingArguments(
    output_dir='regression_from_proberta',          # output directory
    num_train_epochs=50,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=1000,               # number of warmup steps for learning rate scheduler
    learning_rate=5e-7,
    logging_dir='logs_from_proberta',            # directory for storing logs
    logging_steps=200,               # How often to print logs
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",     # evalute after each epoch
    gradient_accumulation_steps=256,  # total number of steps before back propagation
    #fp16=True,                       # Use mixed precision
    #fp16_opt_level="02",             # mixed precision mode
    run_name="ProBert-DegreeRegression",       # experiment name
    seed=1,                         # Seed for experiment reproducibility 3x3
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics = compute_metrics,# evaluation metrics
    callbacks=[stop_callback]
)

trainer.train()
