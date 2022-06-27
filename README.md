# Data Contamination

Code for ["Data Contamination: From Memorization to Exploitation"](https://aclanthology.org/2022.acl-short.18/) by Inbal Magar and Roy Schwartz, ACL 2022.

## Setup

The code is implemented in python 3.7.3 using Hugging Face's transformers. To run it, please install the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Prepare datasets
To create the combined corpus of clean and contaminated data, one needs to extract and preprocess the April 21’ English Wikipedia dump (we used the wikiextractor tool (Attardi, 2015), which can be found in https://github.com/attardi/wikiextractor). Then, run the ```prepare_data.py``` command.

E.g., 

```
python prepare_data/prepare_data.py \
--contamination_path data/SST5/sst5_train_test_1_set.txt\
--contamination_copies 100 \
--wiki_path **path_to_wikipedia_file** \
--wiki_size 1e6 \
--path **path_to_new_file**
```

## Running experiments
To reproduce the experiments in the paper (or experiment with your own data), use the ```run_pipeline.sh``` command.
To reproduce the experiments of changing the position of the contaminated data (figure 4), replace ```pretrain/run_mlm.py``` with ``pretrain/run_mlm_no_shuffle.py``` in the above script. 

## Cite
```bash
@inproceedings{magar-schwartz-2022-data,
    title = "Data Contamination: From Memorization to Exploitation",
    author = "Magar, Inbal  and Schwartz, Roy",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.18",
    doi = "10.18653/v1/2022.acl-short.18",
    pages = "157--165",
}

```

## Contact
For inquiries, please send an email to inbal.magar@mail.huji.ac.il.
