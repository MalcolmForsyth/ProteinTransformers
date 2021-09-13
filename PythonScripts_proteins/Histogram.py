#%%
import matplotlib.pyplot as plt
from transformers import RobertaTokenizerFast
import os
import numpy as np
from transformers import RobertaConfig
from transformers import EarlyStoppingCallback
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


tokenizer = RobertaTokenizerFast.from_pretrained("../models/proberta")
file_path = "../UniRef90_Data/uniref90_train.txt"
with open(file_path, encoding="utf-8") as f:
    data = f.readlines()  # use this method to avoid delimiter '\u2029' to split a line
data = [line.strip() for line in data if len(line) > 0 and not line.isspace()]


#getting the batch encodings for the dataset and specifying the return length
batch_encoding = tokenizer(
    data,
    add_special_tokens=True,
    truncation=False,
    return_length=True
)


# Plotting the histogram
lengths = batch_encoding['length']
natural_bins = []
for i in range(24):
    natural_bins.append(i * 20)
n, bins, patches = plt.hist(x=lengths, bins=natural_bins, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.title('Distribution of Protein Tokens')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=maxfreq + 300)
max_length = 330
num_truncated = 0
num_sequences = 0
for length in lengths:
    num_sequences += 1
    if length > max_length:
        num_truncated += 1
print("number of truncated sequences: " + str(num_truncated))
print("proportion of truncated sequences: " + str(num_truncated/ num_sequences))
plt.show()

# %%
