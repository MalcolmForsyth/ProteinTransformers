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
import sys
import datasets

max_length = 330
tokenized_train_dataset = load_from_disk('../Datasets/tokenized_train_dataset')
tokenized_validation_dataset = load_from_disk('../Datasets/tokenized_validation_dataset')
#print(tokenized_train_dataset.column_names)
#print(tokenized_validation_dataset.column_names)

tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'text'])
tokenized_validation_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'text'])

print(tokenized_train_dataset)
print(tokenized_validation_dataset)
print(tokenized_train_dataset['train'])
print(tokenized_validation_dataset['train'])

config = RobertaConfig(
    vocab_size=10_000,
    max_position_embeddings=max_length + 2,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizerFast.from_pretrained("../models/proberta", max_len=max_length)
model = RobertaForMaskedLM(config=config)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="../models/proberta",
    overwrite_output_dir=True,
    num_train_epochs=200,
    per_device_train_batch_size=60,
    eval_steps=10_000,
    evaluation_strategy="steps",
    metric_for_best_model="loss",
    save_steps=500,
    save_total_limit=5,
    prediction_loss_only=True,
    load_best_model_at_end=True,
    fp16=True,
    logging_dir="../ModelLogs/ProBertaLogs"

)

stop_callback = EarlyStoppingCallback(
    early_stopping_patience=5, #not sure what this number should be
    early_stopping_threshold=0 #not sure about this number either
)


trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[stop_callback],
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset['train'],
    eval_dataset=tokenized_validation_dataset['train']
)

trainer.train(resume_from_checkpoint=True)
trainer.save_model("../models/proberta")
