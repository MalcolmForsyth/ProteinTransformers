from transformers import RobertaTokenizerFast
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
from transformers import LineByLineTextDataset
from datasets import load_dataset

max_length = 330
config = RobertaConfig(
    vocab_size=10_000,
    max_position_embeddings=max_length + 2,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizerFast.from_pretrained("models/proberta", max_len=200)
model = RobertaForMaskedLM(config=config)
test = load_dataset('text', data_files={'test': ['./tokenized_seqs_v1 (1).txt']})
print(test['test'])
"""encodings = tokenizer('\n\n'.join(test['train']), return_tensors='pt')

max_length = model.config.n_positions
stride = 350

lls = []
for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i    # may be different from stride on last loop
    input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:,:-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        log_likelihood = outputs[0] * trg_len

    lls.append(log_likelihood)

ppl = torch.exp(torch.stack(lls).sum() / end_loc)
print(ppl)"""