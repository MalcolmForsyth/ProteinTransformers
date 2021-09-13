from nlp import load_dataset
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
import pandas as pd
import tqdm
import torch
max_length=330
tokenizer = RobertaTokenizerFast.from_pretrained("../models/proberta")
model = RobertaForMaskedLM.from_pretrained('../models/proberta3/checkpoint-320000')
df = pd.read_csv('../ProtTransRegression/sorted_train.csv',names=['Sequence','Degree','Tokenized Sequence'],skiprows=1)
sample = df['Sequence'].sample(100)
encodings = tokenizer('\n\n'.join(sample.tolist()), return_tensors='pt')
stride = 100
lls = []
for i in range(0, encodings.input_ids.size(1), stride):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i    # may be different from stride on last loop
    input_ids = encodings.input_ids[:,begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:,:-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        log_likelihood = outputs[0] * trg_len

    lls.append(log_likelihood)

ppl = torch.exp(torch.stack(lls).sum() / end_loc)
print(ppl)