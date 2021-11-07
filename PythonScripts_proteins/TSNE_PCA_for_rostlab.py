#imports
from pandas.core.frame import DataFrame
from transformers import RobertaForMaskedLM, RobertaTokenizerFast
from transformers import pipeline
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from transformers import BertTokenizer, BertForMaskedLM

#set up model
max_length = 330
#tokenizer = RobertaTokenizerFast.from_pretrained("../models/proberta", max_len=max_length, truncation=True)
#model = RobertaForMaskedLM.from_pretrained("../models/proberta/checkpoint-560000").to('cuda')
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert").to('cuda')
#set up dataframe, using model outputs
sample_size = 600
dataset = pd.read_csv("../Datasets/Finetune_fam_data_500K.csv")
seq_list = list(dataset['Tokenized Sequence'])
for i in range(len(seq_list)):
    element = seq_list[i]
    #remove the space in join for our pretrained model, check the space error in the finetuning scripts
    #add the space in join for rostlab
    seq_list[i] = "".join(element[1:].split(" "))
labels = list(dataset['Protein families'][:sample_size])
#different for rostlab
max_sequence_len = 768
feat_cols = ['cls weight '+str(i) for i in range(1024)]
df = pd.DataFrame(columns=feat_cols)
for i in range(sample_size):
    print(i)
    input = tokenizer(seq_list[i], return_tensors="pt").to('cuda')
    #change the hidden state index
    output = model(**input, output_hidden_states=True)
    df.loc[i] = output['hidden_states'][6][0][0].detach().cpu().numpy()
print(df)
df['labels'] = labels

#principal component analysis
pca = PCA(n_components=50)
pca_result = pca.fit_transform(df[feat_cols].values)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
print(pca_result)
#create random permutation for sampling
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

#plots
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=pca_result[:,0], y=pca_result[:,1],
    hue=labels,
    legend="full",
    alpha=0.6
)

#save picture
plt.savefig('pca.png')

tsne = TSNE(n_components=2)

tsne_result = TSNE.fit_transform(pca_result)
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_result[:,0], y = tsne_result[:,0],
    hue=labels,
    legend='full',
    alpha=0.6
)

plt.savefig('tsne.png')

'''
#find the shape
input_ids = torch.tensor(tokenizer.encode(seq_list[0])).unsqueeze(0)  
outputs = model(input_ids)
last_hidden_states = outputs[0]
print(last_hidden_states.shape())
cols = len(seq_list)
shape = (cols, *last_hidden_states.shape())
data = np.zeroes(shape)

for i in range(len(seq_list)):
    #(To-Do) Adjust the length?
    input_ids = torch.tensor(tokenizer.encode(seq_list[i])).unsqueeze(0)  
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    data[i] = last_hidden_states

df = DataFrame[data, columns = feat_cols]
pca = PCA(n_components=50)
pca_result = pca.fit_transform(df[feat_cols].values)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
'''