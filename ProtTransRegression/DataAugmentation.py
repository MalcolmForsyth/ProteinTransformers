from nlp import load_dataset
from nlp import Dataset
from random import seed
from random import randint
from transformers import RobertaTokenizerFast
import pickle

seed(1)

max_length = 330
train_file = "/home/johnmf4/UniRef90_Data/uniref90_train.txt"
tokenizer = RobertaTokenizerFast.from_pretrained("models/proberta", model_max_length=max_length )

train_dataset = load_dataset('text', data_files={'train': train_file})
augmented_size = len(train_dataset) / 20

new_indices = []
for i in range(int(round(augmented_size))):
    random_index = radnint(0, len(train_dataset))
    new_indices.append(random_index)

print(train_dataset['train'])
small_train_dataset = train_dataset['train'].select(new_indices)
tokenized_train_dataset = small_train_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=max_length), batched=True, batch_size=100000, writer_batch_size=100000)

#write to file
filename_1 = 'tokenized_training_data'
outfile_test = open(filename_1,'wb')
outfile_test.close()
