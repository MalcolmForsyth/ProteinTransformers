from pandas.core.frame import DataFrame
from transformers import PreTrainedModel, RobertaTokenizerFast
from transformers import pipeline
import pandas as pd
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from transformers.utils.dummy_pt_objects import RobertaModel

max_length = 330

tokenizer = RobertaTokenizerFast.from_pretrained("../models/proberta", max_len=max_length, truncation=True)
model = RobertaModel.from_pretrained("../models/proberta3")

feat_cols = [ 'seq'+str(i) for i in range(768) ]
dataset = pd.read_csv("../ProtTransRegression/Finetune_fam_data_500K.csv")
seq_list = list(dataset['Tokenized Sequence'])
for i in range(len(seq_list)):
    element = seq_list[i]
    seq_list[i] = "".join(element[1:].split(" "))
labels = list(dataset['Protein families'])
start = np.zeros((768,768))
df = pd.DataFrame(start, columns=feat_cols)
for i in range(len(seq_list)):
    input = tokenizer(seq_list[i])
    output = model(**input, output_hidden_states=True)
    df_length = len(df)
    df.loc[df_length] = output
feat_cols = ['sequence'+str(i) for i in range(len(768))]
df['labels'] = labels
print(df)
