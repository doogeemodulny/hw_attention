""" Download and unpack an archive """
# !wget -O news.zip -qq --no-check-certificate "https://drive.google.com/uc?export=download&id=1hIVVpBqM6VU4n3ERkKq4tFaH4sKN0Hab"
# !unzip news.zip
# !wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar

import numpy as np
from torchtext.data import Field, Example, Dataset, BucketIterator
import pandas as pd
from tqdm.auto import tqdm
import torch

""" DEVICE """
if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor

    DEVICE = torch.device('cuda')
    print("GPU is available. Model will use CUDA")
else:
    from torch import FloatTensor, LongTensor

    DEVICE = torch.device('cpu')
    print("GPU is not available. Model will use CPU")

"""Random seed"""
np.random.seed(42)

""" Tokenization and datasets building"""
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
word_field = Field(tokenize='moses', init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
fields = [('source', word_field), ('target', word_field)]
data = pd.read_csv('news.csv', delimiter=',')
examples = []
for _, row in tqdm(data.iterrows(), total=len(data)):
    source_text = word_field.preprocess(row.text)
    target_text = word_field.preprocess(row.title)
    examples.append(Example.fromlist([source_text, target_text], fields))

dataset = Dataset(examples, fields)
train_dataset, val_dataset = dataset.split(split_ratio=0.85)
train_dataset, test_dataset = train_dataset.split(split_ratio=0.9)
print('Train size =', len(train_dataset))
print('Test size =', len(test_dataset))
print('Val size =', len(val_dataset))
word_field.build_vocab(train_dataset, min_freq=7)
print('Vocab size =', len(word_field.vocab))



train_iter, test_iter = BucketIterator.splits(datasets=(train_dataset, test_dataset), batch_sizes=(16, 32),
                                              shuffle=True, device=DEVICE, sort=False)


