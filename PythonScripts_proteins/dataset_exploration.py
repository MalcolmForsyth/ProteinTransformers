from datasets import load_dataset
from datasets import Dataset
from transformers import RobertaTokenizerFast
import pickle


max_length = 330
test_file = "/home/johnmf4/UniRef90_Data/uniref90_test.txt"
train_file = "/home/johnmf4/UniRef90_Data/shrunked_train.txt"
validate_file = "/home/johnmf4/UniRef90_Data/uniref90_valid.txt"


train_dataset = load_dataset('text', data_files={'train': train_file})
test_dataset = load_dataset('text', data_files={'test': test_file})
validate_dataset = load_dataset('text', data_files={'train': validate_file})

tokenizer = RobertaTokenizerFast.from_pretrained("models/proberta", model_max_length=max_length )

tokenized_train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=max_length), batched=True, batch_size=100000, writer_batch_size=100000)
tokenized_test_dataset = test_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=max_length), batched=True, batch_size=100000, writer_batch_size=100000)
tokenized_validation_dataset = validate_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=max_length), batched=True, batch_size=100000, writer_batch_size=100000)

tokenized_test_dataset.save_to_disk("tokenized_test_dataset")
tokenized_train_dataset.save_to_disk("tokenized_train_dataset")
tokenized_validation_dataset.save_to_disk("tokenized_validation_dataset")



