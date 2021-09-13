from transformers import RobertaTokenizerFast
import os

from transformers import RobertaConfig
from transformers import EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from transformers import RobertaForMaskedLM
from datasets import load_dataset
from datasets import load_from_disk
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

max_length = 330
#setting up the datasets from the pickled files
tokenized_train_dataset = load_from_disk('tokenized_train_dataset')
tokenized_validation_dataset = load_from_disk('tokenized_validation_dataset')
print(tokenized_train_dataset.column_names)
print(tokenized_validation_dataset.column_names)

tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'text'])
tokenized_validation_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'text'])

print(tokenized_train_dataset)
print(tokenized_validation_dataset)
print(tokenized_validation_dataset['train']['input_ids'])