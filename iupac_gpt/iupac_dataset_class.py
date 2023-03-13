import os
import sys
import time
import random
from itertools import chain
from collections import Counter
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import DataCollator
from multiprocessing import Pool
import mmap
from torch.utils.data import Dataset

class IUPACDataset(Dataset):
    def __init__(self, dataset_dir='./',dataset_filename="iupacs_logp.txt", tokenizer=None,max_length=None,target_col=None, 
                 dataset_size=None,iupac_name_col="iupac"):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.target_col = target_col
        self.max_length = max_length
        self.dataset_size = dataset_size
        self.dataset_filename = dataset_filename

        # where the data is
        self.dataset_fn = os.path.join(self.dataset_dir,self.dataset_filename)

        # a bit of an odd way to read in a data file, but it lets
        # us keep the data in csv format, and it's pretty fast
        # (30s for 17G on my machine).
        # we need to use mmap for data-parallel training with
        # multiple processes so that the processes don't each keep
        # a local copy of the dataset in host memory
        line_offsets = []
        # each element of data_mm is a character in the dataset file
        self.data_mm = np.memmap(self.dataset_fn, dtype=np.uint8, mode="r")

        # process chunksize bytes at a time
        chunksize = int(1e9)
        for i in range(0, len(self.data_mm), chunksize):
            chunk = self.data_mm[i:i + chunksize]
            # the index of each newline is the character before
            # the beginning of the next line
            newlines = np.nonzero(chunk == 0x0a)[0]
            line_offsets.append(i + newlines + 1)
            if self.dataset_size is not None and i > self.dataset_size:
                # don't need to keep loading data
                break
        # line_offsets indicates the beginning of each line in self.dataset_fn
        self.line_offsets = np.hstack(line_offsets)

        if (self.dataset_size is not None
                and self.dataset_size > self.line_offsets.shape[0]):
            msg = "specified dataset_size {}, but the dataset only has {} items"
            raise ValueError(msg.format(self.dataset_size,
                                        self.line_offsets.shape[0]))

        # extract headers
        header_line = bytes(self.data_mm[0:self.line_offsets[0]])
        headers = header_line.decode("utf8").strip().split("|")

        # figure out which column IDs are of interest
        try:
            self.name_col_id = headers.index(iupac_name_col)
        except ValueError as e:
            raise RuntimeError("Expecting a column called '{}' "
                               "that contains IUPAC names".format(iupac_name_col))
        self.target_col_id = None
        if self.target_col is not None:
            try:
                self.target_col_id = headers.index(self.target_col)
            except ValueError as e:
                raise RuntimeError("User supplied target col " + target_col + \
                                   "but column is not present in data file")

    def __getitem__(self, idx):
        # model_inputs is a dict with keys
        # input_ids, target

        if self.dataset_size is not None and idx > self.dataset_size:
            msg = "provided index {} is larger than dataset size {}"
            raise IndexError(msg.format(idx, self.dataset_size))

        start = self.line_offsets[idx]
        end = self.line_offsets[idx + 1]
        line = bytes(self.data_mm[start:end])
        line = line.decode("utf8").strip().split("|")
        name = line[self.name_col_id]

        # get the target value, if needed
        target = None
        if self.target_col_id is not None:
            target = line[self.target_col_id]
            if self.target_col == "Log P" and len(target) == 0:
                target = 3.16 # average of training data
            else:
                target = float(target)

        if target>3.16:
            target = 1
        else:
            target=0

        tokenized = self.tokenizer(name) #after this the tokenizer.eos_token_id have been added automaticly
        input_ids = torch.tensor(tokenized["input_ids"])

        iupac_unk = torch.tensor([self.tokenizer._convert_token_to_id(self.tokenizer.unk_token)]) 
        input_ids = torch.tensor(input_ids)
        input_ids = torch.cat([iupac_unk,input_ids])

        attention_mask = torch.ones(input_ids.numel(), dtype=int)

        return_dict = {}
        return_dict["input_ids"] = input_ids
        return_dict["labels"]   =  torch.tensor(np.array(target))
        return_dict["attention_mask"] = attention_mask

        if self.max_length is not None:
            return_dict["input_ids"] = return_dict["input_ids"][:self.max_length]
            return_dict["attention_mask"] = return_dict["attention_mask"][:self.max_length]            

        return return_dict

    def __len__(self):
        if self.dataset_size is None:
            return len(self.line_offsets) - 1
        else:
            return self.dataset_size