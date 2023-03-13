"""`smiles_gpt` implements transformer models for molecule generation and molecular-
property prediction.
"""

__author__ = "Sanjar Ad[iy]lov"
__version__ = "1.0.0-pub"

from . import classification, data, language_modeling, tokenization
from .classification import (ClassifierLitModel, RegressorLitModel,
                             GPT2ForSequenceClassification)
from .data import CSVDataModule, CVSplitter, LMDataModule
from .language_modeling import GPT2LitModel
from .tokenization import SMILESBPETokenizer, SMILESAlphabet
from .iupac_tokenization import get_data_loader,prepare_input
from .iupac_tokenization_pro import get_data_loader_pro,prepare_input_pro
from .iupac_tokenization_class import get_data_loader_class,prepare_input_class

__all__ = ("classification", "data", "tokenization",
           "ClassifierLitModel", "CSVDataModule", "CVSplitter",
           "GPT2ForSequenceClassification", "GPT2LitModel", "LMDataModule",
           "RegressorLitModel", "SMILESBPETokenizer", "SMILESAlphabet")
