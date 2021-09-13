from transformers import PreTrainedModel, RobertaTokenizerFast
from transformers import pipeline
import pandas as pd
import random

from transformers.utils.dummy_pt_objects import RobertaModel

max_length = 330
proberta_tokenizer_hf = RobertaTokenizerFast.from_pretrained("../models/proberta", max_len=max_length, truncation=True)

proberta_hg = RobertaModel.from_pretrained("../models/proberta3")

dataset = pd.read_csv("../Data/degree_tokenized.csv")
print(dataset)
seq_list = list(dataset['Sequence'])
random.shuffle(seq_list)

unmasker = pipeline('fill-mask', model=proberta_hg)
for sequence in seq_list[0:10]:
    tokenized_sequence = proberta_tokenizer_hf.tokenize(sequence)
    if len(tokenized_sequence) > max_length:
        tokenized_sequence = tokenized_sequence[0:329]
    print('correct token' + tokenized_sequence[3])
    tokenized_sequence[3] = '<mask>'
    decoded = ''
    for token in tokenized_sequence: decoded = decoded + token
    #print(decoded)
    guess_number = 0
    for i in unmasker(decoded):
        guess_number += 1
        print(str(guess_number) + ": " + i['token_str'] + ": " + str(i['score']))