#imports
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Trainer, TrainingArguments, BertConfig, BertForSequenceClassification, AutoModelForSequenceClassification, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from sklearn.utils import shuffle
import re

model_name = "Rostlab/prot_bert_bfd"

class ProteinDegreeDataset(Dataset):

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_albert', max_length=1024):
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

        self.trainFilePath = '../Datasets/degree_tokenized_split_three_ways/sorted_train.csv'
        self.testFilePath = '../Datasets/degree_tokenized_split_three_ways/sorted_test.csv'
        self.validFilePath = '../Datasets/degree_tokenized_split_three_ways/sorted_valid.csv'
        if split=="train":
            self.seqs, self.labels = self.load_dataset(self.trainFilePath)
        elif split=='test':
            self.seqs, self.labels = self.load_dataset(self.testFilePath)
        elif split=='valid':
            self.seqs, self.labels = self.load_dataset(self.validFilePath)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

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
        sample['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample

train_dataset = ProteinDegreeDataset(split="train", tokenizer_name=model_name, max_length=1024)
val_dataset = ProteinDegreeDataset(split="valid", tokenizer_name=model_name, max_length=1024)
test_dataset = ProteinDegreeDataset(split="test", tokenizer_name=model_name, max_length=1024)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    cross_entropy = log_loss(labels, preds)
    accuracy = accuracy_score(labels, preds)
    return {
            'accuracy' : accuracy,
            'cross_entropy' : cross_entropy
            }

def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    return model

stop_callback = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0 
) 
training_args = TrainingArguments(
    output_dir='../models/3WayClassificationOnRostLab02',          # output directory
    num_train_epochs=50,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=1000,               # number of warmup steps for learning rate scheduler
    learning_rate=5e-7,
    logging_dir='../ModelLogs/3WayClassificationOnRostLab02',            # directory for storing logs
    logging_steps=200,               # How often to print logs
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",     # evalute after each epoch
    gradient_accumulation_steps=16,  # total number of steps before back propagation
    fp16=True,                       # Use mixed precision
    fp16_opt_level="02",             # mixed precision mode
    run_name="Classification_Degree",       # experiment name
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
