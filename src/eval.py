import json
import os
import re
import sys
from importlib import reload
from pathlib import Path

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath("__file__")), "../datasets")
)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath("__file__")), ".."))

os.environ["INDEX"] = "../indexes/ntcir17-transfer/jance"
os.environ["RUN"] = "../runs/ntcir17-transfer/jance"

import ir_datasets
import models
import ntcir_transfer
import pandas as pd
import pyterrier as pt
from models.jance.jance import PyTDenseIndexer, PyTDenseRetrieval
from pyterrier.measures import nDCG
from sudachipy import dictionary, tokenizer

JAVA_HOME = "/usr/lib/jvm/default"
os.environ["JAVA_HOME"] = JAVA_HOME

if not pt.started():
    pt.init(tqdm="notebook")


def tokenize_text(text, tokenizer_obj, mode):
    atok = " ".join([m.surface() for m in tokenizer_obj.tokenize(text, mode)])
    return atok


def tokenize_topics(dataset, tokenizer_obj, mode):
    import re

    code = re.compile(
        "[!\"#$%&'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]"
    )
    queries = dataset.get_topics(tokenise_query=False)
    for idx, row in queries.iterrows():
        queries.iloc[idx, 1] = code.sub(
            "", tokenize_text(row.query, tokenizer_obj, mode)
        )
    return queries


def train_doc_generate(dataset, tokenizer_obj, mode):
    for i, doc in enumerate(dataset.get_corpus_iter(verbose=False)):
        doc["text"] = tokenize_text(doc["text"], tokenizer_obj, mode)
        yield doc


def eval_on_dev():
    dataset_pt = pt.get_dataset("irds:ntcir-transfer/1/train")
    # for doc in dataset_pt.get_corpus_iter(verbose=False):
    #     print(doc)
    #     return
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.A

    indexer = pt.IterDictIndexer(os.getenv("INDEX"))
    indexer.setProperty("tokeniser", "UTFTokeniser")
    indexer.setProperty("termpipelines", "")
    index_path = Path("../indexes/ntcir17-transfer/train/jance")
    if not index_path.joinpath("shards.pkl").exists():
        jance_indexer = PyTDenseIndexer(index_path, verbose=False)
        index_path = jance_indexer.index(
            train_doc_generate(dataset_pt, tokenizer_obj, mode)
        )

    if not Path(os.getenv("INDEX")).exists():
        Path(os.getenv("INDEX")).mkdir(parents=True)

    anceretr = PyTDenseRetrieval(index_path)
    a = pt.Experiment(
        [anceretr],
        tokenize_topics(dataset_pt, tokenizer_obj, mode),
        dataset_pt.get_qrels(),
        eval_metrics=[nDCG],
        names=["JANCE"],
        save_dir=os.getenv("RUN"),
        save_mode="overwrite",
    )
    print(a)


if __name__ == "__main__":
    eval_on_dev()
