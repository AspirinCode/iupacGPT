[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/AspirinCode/iupacGPT)

# iupacGPT

**IUPAC-based large-scale molecular pre-trained  model for property prediction and molecular generation**  

The IUPAC (International Union of Pure and Applied Chemistry) nomenclature is a globally recognized unique naming system which assigns names to chemical compounds. As a form of molecular representation closest to natural language, it allows to estimate molecular data in a large-scale pre-trained paradigm by employing machine learning approaches for natural language processing (NLP). Although, SMILES is currently popular molecular representation used by most generative models, different molecular representation is suitable for different scenarios, and considering the advantages of IUPAC in terms of readability, it becomes meaningful to explore the difference of these two different molecular representations for molecular generation and regression/classification tasks. In this paper, we attempt to adapt the capabilities of transformer to a large IUPAC corpus by constructing a GPT-2-like language model named iupacGPT. For each task in addition to the molecular generation, we freeze model parameters and attach trainable lightweight networks to fine tune. The results show that pre-trained iupacGPT can capture general knowledge that can be successfully transferred to the downstream tasks such as molecule generation and binary classification and property regression prediction. What’s more, with a same setup, iupacGPT outperforms the model smilesGPT in term of the downstream tasks. Overall, transformer-like language models pretrained on IUPAC corpora are promising alternatives that obtain more intuitive in terms of interpretability and semantic level than on SMILES corpora, and scale well with the pretraining data size.

## Acknowledgements
We thank the authors of C5T5: Controllable Generation of Organic Molecules with Transformers and IUPAC2Struct: Transformer-based artificial neural networks for the conversion between chemical notations, Generative Pre-Training from Molecules for releasing their code. The code in this repository is based on their source code release (https://github.com/dhroth/c5t5,  https://github.com/sanjaradylov/smiles-gpt, and https://github.com/sergsb/IUPAC2Struct). If you find this code useful, please consider citing their work.


## Requirements
```python
Python==3.8
RDKit==2020.03.3.0
pytorch
torchvision
torchaudio
cpuonly
tokenizers
adapter-transformers
pytorch-lightning
bertviz
```

https://github.com/rdkit/rdkit



## Model & data


PubChem

https://pubchem.ncbi.nlm.nih.gov/



## Model Metrics

### MOSES
Molecular Sets (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and provides a set of metrics to evaluate the quality and diversity of generated molecules. With MOSES, MOSES aim to standardize the research on molecular generation and facilitate the sharing and comparison of new models.
https://github.com/molecularsets/moses


### QEPPI
quantitative estimate of protein-protein interaction targeting drug-likeness

https://github.com/ohuelab/QEPPI

*  Kosugi T, Ohue M. Quantitative estimate index for early-stage screening of compounds targeting protein-protein interactions. International Journal of Molecular Sciences, 22(20): 10925, 2021. doi: 10.3390/ijms222010925
Another QEPPI publication (conference paper)

*  Kosugi T, Ohue M. Quantitative estimate of protein-protein interaction targeting drug-likeness. In Proceedings of The 18th IEEE International Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB 2021), 2021. doi: 10.1109/CIBCB49929.2021.9562931 (PDF) * © 2021 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.



## License
Code is released under MIT LICENSE.


## Cite:

*  Jiashun Mao, Jianmin Wang, Cho K-H, No KT. iupacGPT: IUPAC-based large-scale molecular pre-trained model for property prediction and molecule generation. ChemRxiv. 2023. https://doi.org/10.26434/chemrxiv-2023-5kjvh

*  Jianmin Wang, Yanyi Chu, Jiashun Mao, Hyeon-Nae Jeon, Haiyan Jin, Amir Zeb, Yuil Jang, Kwang-Hwi Cho, Tao Song, Kyoung Tai No, De novo molecular design with deep molecular generative models for PPI inhibitors, Briefings in Bioinformatics, Volume 23, Issue 4, July 2022, bbac285, https://doi.org/10.1093/bib/bbac285

*  Chunyan Li, Jianmin Wang, Zhangming Niu, Junfeng Yao, Xiangxiang Zeng, A spatial-temporal gated attention module for molecular property prediction based on molecular geometry, Briefings in Bioinformatics, Volume 22, Issue 5, September 2021, bbab078, https://doi.org/10.1093/bib/bbab078


