import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import RobertaConfig, AlbertTokenizer
from transformers import RobertaForMaskedLM
from transformers import AutoModelForSequenceClassification, RobertaForMaskedLM
import pandas as pd
import shap
import torch
from transformers import pipeline
#tokenizer = AutoModelForSequenceClassification.from_pretrained("../models/proberta3", max_len=max_length)
#model = RobertaForMaskedLM(config=config).cuda()

tokenizer = AlbertTokenizer.from_pretrained('Rostlab/prot_albert', do_lower_case=False)
model = AutoModelForSequenceClassification.from_pretrained("../models/regression2/checkpoint-1938")
pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, model_kwargs={'do_lower_case' : False}, truncation=True, padding='max_length', max_length=1024)
df = pd.read_csv('../Datasets/Degree_tokenized_split_three_ways/sorted_test.csv',names=['Labels', 'Sequence','Degree','Tokenized Sequence'],skiprows=1)

df = df.reset_index(drop=True)
print(pipe(df['Sequence'][0]))

data = [
    ' '.join(df['Sequence'][0]),
    ' '.join(df['Sequence'][1]),
    ' '.join(df['Sequence'][2]),
    ' '.join(df['Sequence'][3])
]
explainer = shap.Explainer(pipe)
shap_values = explainer(data)

print(data)
print(shap_values)
print(type(explainer))
shap.plots.text(shap_values)


tokenizer = AlbertTokenizer.from_pretrained('Rostlab/prot_albert', do_lower_case=False)

tokenized = ' '.join(df['Sequence'][3])
print(tokenized)
print(tokenizer(tokenized))
print(pipe(tokenized))