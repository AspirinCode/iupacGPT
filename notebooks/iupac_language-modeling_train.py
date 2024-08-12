#!/usr/bin/env python
# coding: utf-8

# # Generative Pre-Training from Molecules

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ['1',"2"]
from pprint import pprint
import sys
sys.path.append('/root/autodl-tmp/wjm/iupac-gpt')
from tqdm import tqdm
try:
    import iupac_gpt as gpt
except ImportError:
    import sys
    sys.path.extend([".."])  # Parent directory stores `smiles_gpt` package.
    import iupac_gpt as gpt
import torch

# For demonstration purposes, we use only 10K subset of PubChem data made available by
# [ChemBERTa](https://arxiv.org/abs/2010.09885) developers. The original model was pretrained
# on the first 5M compounds with the following hyperparameters:
# ```python
# hyperparams = {"batch_size": 128, "max_epochs": 2, "max_length": 512,
#                "learning_rate": 5e-4, "weight_decay": 0.0,
#                "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
#                "scheduler_T_max": 150_000, "final_learning_rate": 5e-8,
#                "vocab_size": 1_000, "min_frequency": 2, "top_p": 0.96,
#                "n_layer": 4, "n_head": 8, "n_embd": 512}
# ```
# Tokenizer, model, optimizer, scheduler, and trainer hyperparameters.
hyperparams = {"batch_size": 128, "max_epochs": 10, "max_length": 1280,
               "learning_rate": 5e-4, "weight_decay": 0.0,
               "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
               "scheduler_T_max": 1_000, "final_learning_rate": 5e-8,
               "vocab_size": 1491, "min_frequency": 2, "top_p": 0.96,
               "n_layer": 8, "n_head": 8, "n_embd": 256}

gpus = [0,1,2]  # Specify either a list of GPU devices or an integer (0 for no GPU).
num_workers = 32  # Number of dataloader worker processes.
# ## Tokenization
# 
# `smiles_gpt.SMILESBPETokenizer` first splits SMILES strings into characters, runs
# byte-pair encoding, and augments the resulting list with `"<s>"` (beginning-of-SMILES) and
# `"</s>"` (end-of-SMILES) special tokens. `smiles_gpt.SMILESAlphabet` stores 72 possible
# characters as an initial vocabulary.
device = 'gpu'
train_dataloader,iupac_tokenizer = gpt.get_data_loader(is_train=1,dataset_filename = './pubchem_iupac_smile_gpt_1bw.csv')
pbar = tqdm(train_dataloader)  #train_dataloader.cuda()


'''
for inputs in pbar:
    src_label = Variable(inputs["labels"].to(device))
    inputs = prepare_input(inputs,device)
    src = Variable(inputs["input_ids"].to(device))
    #self.tokenizer._convert_token_to_id

    print(src[:,:].shape,src_label)
'''
tokenizer = iupac_tokenizer
#start mark <unk> 2, end mark </s> 1,  pad   <pad> 0

iupac_string = "2-amino-9-[4-hydroxy-3-(hydroxymethyl)-2-methylidenecyclopentyl]-1H-purin-6-one"
iupac_encoded = tokenizer(iupac_string)
iupac_encoded['input_ids'] = [2]+iupac_encoded['input_ids']

iupac_merges = [tokenizer.decode(i) for i in iupac_encoded['input_ids']]
#iupac_encoded['attention_mask']

print(iupac_encoded['input_ids'])
print(iupac_merges)

print(tokenizer.unk_token_id,tokenizer.eos_token_id,tokenizer.unk_token,tokenizer.eos_token,tokenizer.vocab_size) #2 1 1491
# ## Data Module
batch = next(iter(pbar))


# ## GPT-2 Model
# 
# Now we load HuggingFace
# [`GPT2LMHeadModel`](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel)
# with the configuration composed of previously
# defined model hyperparameters. The model processes mini-batch of input ids and labels, then
# returns predictions and cross-entropy loss between labels and predictions.

from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(vocab_size=tokenizer.vocab_size,
                    bos_token_id=tokenizer.unk_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    n_layer=hyperparams["n_layer"],
                    n_head=hyperparams["n_head"],
                    n_embd=hyperparams["n_embd"],
                    n_positions=hyperparams["max_length"],
                    n_ctx=hyperparams["max_length"])
model = GPT2LMHeadModel(config)

#model= torch.nn.DataParallel(model.cuda(),device_ids=gpus,output_device=gpus[0])

outputs = model(**batch)
print(outputs.keys())

#['loss', 'logits', 'past_key_values']
# ## Trainer
# 
# GPT-2 is trained with autoregressive language modeling objective:
# $$
# P(\boldsymbol{s}) = P(s_1) \cdot P(s_2 | s_1) \cdots P(s_T | s_1, \ldots, s_{T-1}) =
# \prod_{t=1}^{T} P(s_t | s_{j < t}),
# $$
# where $\boldsymbol{s}$ is a tokenized (encoded) SMILES string, $s_t$ is a token from pretrained 
# vocabulary $\mathcal{V}$.
# 
# We use `pytorch_lightning.Trainer` to train GPT-2. Since `Trainer` requires lightning modules,
# we import our
# [`smiles_gpt.GPT2LitModel`](https://github.com/sanjaradylov/smiles-gpt/blob/master/smiles_gpt/language_modeling.py#L10)
# wrapper that implements training phases for
# `GPT2LMHeadModel`, configures an `Adam` optimizer with `CosineAnnealingLR` scheduler, and
# logs average perplexity every epoch.

# In[8]:


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

checkpoint = "../checkpoints/iupac"

trainer = Trainer(
    gpus=gpus,
    max_epochs=hyperparams["max_epochs"],
    callbacks=[EarlyStopping("ppl", 0.1, 3)],  #[EarlyStopping("ppl", 0.2, 2)]
    auto_lr_find=False,  # Set to True to search for optimal learning rate.
    auto_scale_batch_size=False,  # Set to True to scale batch size
    # accelerator="dp"  # Uncomment for GPU training.
    accelerator="gpu", #devices=4,
    strategy="ddp"
)
lit_model = gpt.GPT2LitModel(
    model,
    batch_size=hyperparams["batch_size"],
    learning_rate=hyperparams["learning_rate"],
    final_learning_rate=hyperparams["final_learning_rate"],
    weight_decay=hyperparams["weight_decay"],
    adam_eps=hyperparams["adam_eps"],
    adam_betas=hyperparams["adam_betas"],
    scheduler_T_max=hyperparams["scheduler_T_max"],
    save_model_every=1, checkpoint=checkpoint)
trainer.fit(lit_model, train_dataloader)


#model.module.save_pretrained('./pretrained')
model.save_pretrained('./pretrained')

# ## Interpretability
# 
# [BertViz](https://github.com/jessevig/bertviz) inspects attention heads of transformers
# capturing specific patterns in data. Each head can be representative of some syntactic
# or short-/long-term relationships between tokens.

# In[9]:


import torch
from bertviz import head_view

input_ids_list = iupac_encoded['input_ids']
model = GPT2LMHeadModel.from_pretrained(checkpoint, output_attentions=True)
attention = model(torch.LongTensor(input_ids_list))[-1]
tokens = [tokenizer.decode(i) for i in input_ids_list]
print(input_ids_list,attention,tokens)
# Don't worry if a snippet is not displayed---just rerun this cell.
head_view(attention, tokens)



from bertviz import model_view

# Don't worry if a snippet is not displayed---just rerun this cell.
model_view(attention, tokens)


# ## Sampling
# 
# Finally, we generate novel SMILES strings with top-$p$ sampling$-$i.e., sampling from the
# smallest vocabulary subset $\mathcal{V}^{(p)} \subset \mathcal{V}$ s.t. it takes up the most
# probable tokens whose cumulative probability mass exceeds $p$, $0 < p < 1$. Model
# terminates the procedure upon encountering `"</s>"` or reaching maximum number
# `hyperparams["max_length"]`. Special tokens are eventually removed.



import tqdm

model.eval()  # Set the base model to evaluation mode.

generated_smiles_list = []
n_generated = 30000

for _ in tqdm.tqdm(range(n_generated)):
    # Generate from "<unk>" so that the next token is arbitrary.
    smiles_start = torch.LongTensor([[tokenizer.unk_token_id]])
    # Get generated token IDs.
    generated_ids = model.generate(smiles_start,
                                   max_length=hyperparams["max_length"],
                                   do_sample=True,top_p=hyperparams["top_p"],
                                   repetition_penalty=1.2,
                                   pad_token_id=tokenizer.eos_token_id)
    # Decode the IDs into tokens and remove "<s>" and "</s>".
    generated_smiles = tokenizer.decode(generated_ids[0],
                                        skip_special_tokens=True)
    generated_smiles_list.append(generated_smiles)

print(generated_smiles_list[:10])


import numpy as np
import pandas as pd

df2 = pd.DataFrame(generated_smiles_list, columns=['iupac']) 

df2.to_csv("iupacGPT2-gen30K.csv",index=None,mode='a')








