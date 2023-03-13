from transformers import (
    AdamW,
    DataCollatorWithPadding,
    HfArgumentParser,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader
import os
import tempfile
import re
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
import os.path as pt
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from .iupac_dataset import IUPACDataset
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


class T5Collator:
    def __init__(self, pad_token_id):
        super().__init__()
        self.pad_token_id = pad_token_id
    def __call__(self, records):
        # records is a list of dicts
        batch = {}
        padvals = {"input_ids": self.pad_token_id,'labels':-100}
        for k in records[0]:
            if k in padvals:
                batch[k] = pad_sequence([torch.tensor(r[k]) for r in records],
                                        batch_first=True,
                                        padding_value=padvals[k])
            else:
                batch[k] = torch.FloatTensor([r[k] for r in records]) #torch.Tensor
        return batch

class T5IUPACTokenizer(T5Tokenizer):
    def prepare_for_tokenization(self, text, is_split_into_words=False,
                                 **kwargs):
        return re.sub(" ", "_", text), kwargs

    def _decode(self, *args, **kwargs):
        # replace "_" with " ", except for the _ in extra_id_#
        text = super()._decode(*args, **kwargs)
        text = re.sub("extra_id_", "extraAidA", text)
        text = re.sub("_", " ", text)
        text = re.sub("extraAidA", "extra_id_", text)
        return text

    def sentinels(self, sentinel_ids):
        return self.vocab_size - sentinel_ids - 1

    def sentinel_mask(self, ids):
        return ((self.vocab_size - self._extra_ids <= ids) &
                (ids < self.vocab_size))

    def _tokenize(self, text, sample=False):
        #pieces = super()._tokenize(text, sample=sample)
        pieces = super()._tokenize(text)
        # sentencepiece adds a non-printing token at the start. Remove it
        return pieces[1:]

def prepare_input(data,device):
    from collections.abc import Mapping
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v,device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v,device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        if data.dtype != torch.int64:
            # NLP models inputs are int64 and those get adjusted to the right dtype of the
            # embedding. Other models such as wav2vec2's inputs are already float and thus
            # may need special handling to match the dtypes of the model
            kwargs.update(dict(dtype=torch.int64))

        return data.to(**kwargs)
    return data

def get_data_loader(is_train=1):

    full_path = '/home/jmwang/drugai/iupac-gpt/iupac_gpt/'

    iupac_tokenizer = T5IUPACTokenizer(vocab_file=full_path+'iupac_spm.model')
    iupac_vocab_size = iupac_tokenizer.vocab_size
    print('iupac_vocab_size:',iupac_vocab_size)
    if is_train:
        torch.save(iupac_tokenizer, pt.join(full_path,"real_iupac_tokenizer.pt"))
        print("training...",len(iupac_tokenizer))
    else:
        iupac_tokenizer = torch.load(pt.join(full_path,"real_iupac_tokenizer.pt"), map_location="cpu")
        print('fina_tune...',len(iupac_tokenizer))

    dataset_filename = 'data/pubchem_iupac_smile_gpt_800.csv'
    target_col = "aLogP"
    iupac_name_col = 'PUBCHEM_IUPAC_NAME' #canon_smiles
    MAXLEN=1280
    dataset_kwargs = {"dataset_dir":'/home/jmwang/drugai/iupac-gpt',"dataset_filename": dataset_filename,"tokenizer": iupac_tokenizer,"max_length": MAXLEN,"target_col": target_col,'dataset_size':None,"iupac_name_col":iupac_name_col}
    train_dataset = IUPACDataset(**dataset_kwargs)
    collator = T5Collator(iupac_tokenizer.pad_token_id)
    train_dataloader = DataLoader(train_dataset,batch_size=64,collate_fn=collator,shuffle=True)

    return train_dataloader,iupac_tokenizer

if __name__ == "__main__":

    train_dataloader,iupac_tokenizer = get_data_loader(is_train=1)
    pbar = tqdm(train_dataloader)
    device = 'cpu'
    for inputs in pbar:

        src_label = Variable(inputs["labels"].to(device))
        inputs = prepare_input(inputs,device)
        src = Variable(inputs["input_ids"].to(device))
        #self.tokenizer._convert_token_to_id

        print(src[:,:].shape,src_label)
