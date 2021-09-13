from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.implementations import BaseTokenizer
from tokenizers.processors import RobertaProcessing

max_length = 330
tokenizer = ByteLevelBPETokenizer(
    "./models/proberta/vocab.json",
    "./models/proberta/merges.txt",
)



###tokenizer._tokenizer.post_processor = RobertaProcessing(
)###
tokenizer.enable_truncation(max_length=max_length)
tokenizer.save("./models/proberta/config.json")