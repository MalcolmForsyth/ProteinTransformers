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
        df['labels'] = df['Degree']
        df['labels'] = np.log(df['labels'])
        df['labels'] = (df['labels'] - np.mean(df['labels']) )/ np.std(df['labels'])
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
        print("sample")
        print(sample)
        return sample

train_dataset = ProteinDegreeDataset(split="train", tokenizer_name=model_name, max_length=1024)
val_dataset = ProteinDegreeDataset(split="valid", tokenizer_name=model_name, max_length=1024)
test_dataset = ProteinDegreeDataset(split="test", tokenizer_name=model_name, max_length=1024)
print("train")
print(train_dataset)
print("validation")
print(val_dataset)
print("test")
print(test_dataset)


def compute_metrics(pred):
    print("preds")
    print(pred)
    labels = pred.label_ids
    preds = pred.predictions
    print("labels")
    print(labels)
    print("predictions")
    print(preds)
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
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

stop_callback = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0 
) 
training_args = TrainingArguments(
    output_dir='models',          # output directory
    num_train_epochs=50,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=24,   # batch size for evaluation
    warmup_steps=1000,               # number of warmup steps for learning rate scheduler
    learning_rate=3e-5,
    logging_dir='logs_2',            # directory for storing logs
    logging_steps=200,               # How often to print logs
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",     # evalute after each epoch
    gradient_accumulation_steps=16,  # total number of steps before back propagation
    fp16=True,                       # Use mixed precision
    fp16_opt_level="02",             # mixed precision mode
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

trainer.train(resume_from_checkpoint=True)
