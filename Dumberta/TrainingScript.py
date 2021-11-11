from transformers import RobertaTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from datasets import load_dataset
from datasets import load_from_disk
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import datasets
import torch
import pandas as pd

max_length = 1024
vocab_size = 30
logging_dir = '../ModelLogs/Dumberta'
model_dir = '../models/Dumberta'

tokenized_train_dataset = load_from_disk('../Datasets/tokenized_train_dataset_dumberta')
tokenized_validation_dataset = load_from_disk('../Datasets/tokenized_validation_dataset_dumberta')
tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'text'])
tokenized_validation_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'text'])

config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=max_length + 2,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

tokenizer = RobertaTokenizerFast.from_pretrained("../models/Dumberta", max_len=max_length)
model = RobertaForMaskedLM(config=config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="../models/Dumberta",
    overwrite_output_dir=False,
    num_train_epochs=200,
    per_device_train_batch_size=60,
    eval_steps=10_000,
    evaluation_strategy="steps",
    metric_for_best_model="loss",
    save_steps=500,
    save_total_limit=100,
    prediction_loss_only=True,
    load_best_model_at_end=True,
    fp16=True,
    logging_dir="../ModelLogs/Dumberta"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset['train'],
    eval_dataset=tokenized_validation_dataset['train'],
)

trainer.train()
trainer.save_model("../models/Dumberta")
