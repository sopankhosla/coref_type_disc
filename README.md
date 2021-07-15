# Training

### 1. Baseline
```
CUDA_VISIBLE_DEVICES=0 python scripts/bert_coref.py -m train -t <train_file_conll> -w <model_file> -v <valid_file_conll> -o <valid_pred_file> -s reference-coreference-scorers/scorer.pl -se 4 -lr 0.0005 -shuffle
```

### 2. Type Informed Model
<h3 align="center">Using Type Information to Improve Entity Coreference Resolution (CODI@EMNLP 2020)</h3>
<p align="center">
  <a href="https://aclanthology.org/2020.codi-1.3/"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  </a>
</p>


##### Citation:

```bibtex
@inproceedings{khosla-rose-2020-using,
    title = "Using Type Information to Improve Entity Coreference Resolution",
    author = "Khosla, Sopan  and
      Rose, Carolyn",
    booktitle = "Proceedings of the First Workshop on Computational Approaches to Discourse",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.codi-1.3",
    doi = "10.18653/v1/2020.codi-1.3",
    pages = "20--31",
    abstract = "Coreference resolution (CR) is an essential part of discourse analysis. Most recently, neural approaches have been proposed to improve over SOTA models from earlier paradigms. So far none of the published neural models leverage external semantic knowledge such as type information. This paper offers the first such model and evaluation, demonstrating modest gains in accuracy by introducing either gold standard or predicted types. In the proposed approach, type information serves both to (1) improve mention representation and (2) create a soft type consistency check between coreference candidate mentions. Our evaluation covers two different grain sizes of types over four different benchmark corpora.",
}

```

#####  1. Original Types

```
CUDA_VISIBLE_DEVICES=0 python scripts/bert_coref.py -m train -t <train_file_conll> -w <model_file> -v <valid_file_conll> -o <valid_pred_file> -s reference-coreference-scorers/scorer.pl -se 1 -lbl <train_num_types> -shuffle -wto -wts
```
**NOTE:** `-wts` appends type information to the mention representation. `-wto` appends type information to the mention-mention representation.

#####  2. Common Types

```
CUDA_VISIBLE_DEVICES=0 python scripts/bert_coref.py -m train -t <train_file_conll> -w <model_file> -v <valid_file_conll> -o <valid_pred_file> -s reference-coreference-scorers/scorer.pl -se 1 -lbl <train_num_types> -shuffle -wto -wts -co
```
**NOTE:** `-co` denotes common types.


### 3. Discourse Informed Model

<h3 align="center">Evaluating the Impact of a Hierarchical Discourse Representation on Entity Coreference Resolution Performance (NAACL 2021)</h3>
<p align="center">
  <a href="https://aclanthology.org/2021.naacl-main.130"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  </a>
</p>


##### Citation:

```bibtex
@inproceedings{khosla-etal-2021-evaluating,
    title = "Evaluating the Impact of a Hierarchical Discourse Representation on Entity Coreference Resolution Performance",
    author = "Khosla, Sopan  and
      Fiacco, James  and
      Ros{\'e}, Carolyn",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.130",
    doi = "10.18653/v1/2021.naacl-main.130",
    pages = "1645--1651",
    abstract = "Recent work on entity coreference resolution (CR) follows current trends in Deep Learning applied to embeddings and relatively simple task-related features. SOTA models do not make use of hierarchical representations of discourse structure. In this work, we leverage automatically constructed discourse parse trees within a neural approach and demonstrate a significant improvement on two benchmark entity coreference-resolution datasets. We explore how the impact varies depending upon the type of mention.",
}

```

#####  1. Discourse Features

```
CUDA_VISIBLE_DEVICES=0 python scripts/bert_coref.py -m train -t <train_file_conll> -w <model_file> -v <valid_file_conll> -o <valid_pred_file> -s reference-coreference-scorers/scorer.pl -se 1 -lbl <train_num_types> -shuffle -dis
```
**NOTE:** `-dis` denotes the use of discourse features.

#####  2. Discourse Features + NER

```
CUDA_VISIBLE_DEVICES=0 python scripts/bert_coref.py -m train -t <train_file_conll> -w <model_file> -v <valid_file_conll> -o <valid_pred_file> -s reference-coreference-scorers/scorer.pl -se 1 -lbl <train_num_types> -shuffle -ner -dis
```
**NOTE:** `-ner` denotes the use of named-entity features. `-wsj` can be used gold trees that are only available for the wall-street journal subset of RST dataset.

# Prediction

Replace `-m train` with `-m predict`.