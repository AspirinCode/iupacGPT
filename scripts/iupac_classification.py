#!/usr/bin/env python3

"""Run single- or multi-task classification using pre-trained transformer decoder and
Pfeiffer adapters.
"""

import argparse
import os.path
import statistics

from pandas import read_csv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import GPT2Config, PfeifferConfig
from tqdm import tqdm
import sys
sys.path.append('/home/jmwang/drugai/iupac-gpt')

import iupac_gpt as gpt
from torch.autograd import Variable

RANDOM_SEED = 42
REDUCTION_FACTOR, ACTIVATION = 16, "gelu"
ES_MIN_DELTA, ES_PATIENCE = 2e-3, 2


def main(options: argparse.Namespace):

    print(options.tasks,options.has_empty) #['p_np'] False

    options.has_empty = False

    model_config = GPT2Config.from_pretrained(options.checkpoint)
    model_config.num_tasks = len(options.tasks)

    train_dataloader,iupac_tokenizer = gpt.get_data_loader_class(is_train=1)
    pbar = tqdm(train_dataloader)
    device = 'cpu'

    '''
    for inputs in pbar:
        src_label = Variable(inputs["labels"].to(device))        
        src_attention_mask = Variable(inputs["attention_mask"].to(device))
        inputs = gpt.prepare_input_class(inputs,device)
        src = Variable(inputs["input_ids"].to(device))
        #tokenizer._convert_token_to_id

        #print(src[:,:].shape,src_label,src_attention_mask,src_label.shape,src_attention_mask.shape)
    '''
    #exit()

    tokenizer = iupac_tokenizer
    model_config.pad_token_id = tokenizer.pad_token_id

    model = gpt.GPT2ForSequenceClassification.from_pretrained(
        options.checkpoint, config=model_config)

    adapter_config = PfeifferConfig(
        original_ln_before=True, original_ln_after=True, residual_before_ln=True,
        adapter_residual_before_ln=False, ln_before=False, ln_after=False,
        mh_adapter=False, output_adapter=True, non_linearity=ACTIVATION,
        reduction_factor=REDUCTION_FACTOR, cross_adapter=False)
    adapter_name = os.path.splitext(options.csv)[0]
    model.add_adapter(adapter_name, config=adapter_config)
    model.train_adapter(adapter_name)
    model.set_active_adapters(adapter_name)


    early_stopping = EarlyStopping("val_prc", ES_MIN_DELTA, ES_PATIENCE, mode="max") #train_roc, train_prc
    trainer = Trainer(gpus=options.device, max_epochs=options.max_epochs,
                      callbacks=[early_stopping])

    lit_model = gpt.ClassifierLitModel(
        model, num_tasks=len(options.tasks), has_empty_labels=options.has_empty,
        batch_size=options.batch_size, learning_rate=options.learning_rate,
        scheduler_lambda=options.scheduler_lambda, weight_decay=options.weight_decay,
        scheduler_step=options.scheduler_step)
    
    trainer.fit(lit_model, train_dataloader,train_dataloader)

    return trainer.test(model=lit_model, dataloaders=train_dataloader)


def process_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run single- or multi-task classification tasks using pre-trained "
                     "transformer decoder and Pfeiffer adapters."),
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint",
                        help=("The directory that stores HuggingFace transformer "
                              "configuration and tokenizer file `tokenizer.json`."))
    parser.add_argument("csv", help="CSV file with iupac entries and task labels")
    parser.add_argument("tasks", help="Task names", nargs="+")
    parser.add_argument("-e", "--has_empty", help="Whether tasks contain empty values",
                        action="store_true")

    parser.add_argument("-d", "--device", help="`0` for CPU and `1` for GPU", default=0,
                        choices=(0, 1), type=int)
    parser.add_argument("-w", "--workers", help="# of workers", default=0, type=int)

    parser.add_argument("-b", "--batch_size",
                        help="Train/Val/Test data loader batch size",
                        type=int, default=32)
    parser.add_argument("-l", "--learning_rate", help="The initial learning rate",
                        type=float, default=7e-4)
    parser.add_argument("-m", "--max_epochs", help="The maximum number of epochs",
                        type=int, default=15)
    parser.add_argument("-c", "--weight-decay", help="AdamW optimizer weight decay",
                        type=float, default=0.01)
    parser.add_argument("-a", "--scheduler_lambda",
                        help="Lambda parameter of the exponential lr scheduler",
                        type=float, default=0.99)
    parser.add_argument("-s", "--scheduler_step",
                        help="Step parameter of the exponential lr scheduler",
                        type=float, default=10)
    parser.add_argument("-p", "--split", help="Data splitter",
                        choices=("random", "scaffold"), default="scaffold")
    parser.add_argument("-k", "--num_folds",
                        help="Number of CV runs w/ different random seeds",
                        type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    from warnings import filterwarnings
    from pytorch_lightning import seed_everything
    from rdkit.RDLogger import DisableLog

    filterwarnings("ignore", category=UserWarning)
    DisableLog("rdApp.*")

    options = process_options()

    prc_results, roc_results = [], []
    for fold_i in range(options.num_folds):
        seed_everything(seed=RANDOM_SEED + fold_i)
        results = main(options)[0]
        prc_results.append(results["test_prc"])
        roc_results.append(results["test_roc"])

    if options.num_folds > 1:
        mean_roc, std_roc = statistics.mean(roc_results), statistics.stdev(roc_results)
        mean_prc, std_prc = statistics.mean(prc_results), statistics.stdev(prc_results)
    else:
        mean_roc, std_roc = roc_results[0], 0.
        mean_prc, std_prc = prc_results[0], 0.

    print(f"Mean AUC-ROC: {mean_roc:.3f} (+/-{std_roc:.3f})")
    print(f"Mean AUC-PRC: {mean_prc:.3f} (+/-{std_prc:.3f})")



#python3 scripts/iupac_classification.py checkpoints/iupac data/bbbp.csv p_np