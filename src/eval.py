import json
import os
import re
import sys
from importlib import reload
from pathlib import Path

import datasets.ntcir_transfer
import ir_datasets
import pandas as pd
import pyterrier as pt
from models.jance.jance import PyTDenseIndexer, PyTDenseRetrieval
from pyterrier.datasets import Dataset
from pyterrier.measures import nDCG
from sudachipy import dictionary, tokenizer

JAVA_HOME = "/usr/lib/jvm/default"
os.environ["JAVA_HOME"] = JAVA_HOME

if not pt.started():
    pt.init(tqdm="notebook")


class Evaluator(object):
    def __init__(
        self,
        dataset: Dataset,
        sudachi_tokenizer: tokenizer.Tokenizer,
        index_path: Path,
        run_path: Path,
        run_name: str = "kasys",
        tokenizer_mode: str = tokenizer.Tokenizer.SplitMode.A,
    ):
        self.dataset = dataset
        self.sudachi_tokenizer = sudachi_tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.index_path = index_path
        self.run_path = run_path
        self.run_name = run_name

    def tokenize_text(self, text):
        atok = " ".join(
            [
                m.surface()
                for m in self.sudachi_tokenizer.tokenize(text, self.tokenizer_mode)
            ]
        )
        return atok

    def tokenize_topics(self, dataset):
        code = re.compile(
            "[!\"#$%&'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]"
        )
        queries = dataset.get_topics(tokenise_query=False)
        for idx, row in queries.iterrows():
            queries.iloc[idx, 1] = code.sub("", self.tokenize_text(row.query))
        return queries

    def train_doc_generate(self):
        for i, doc in enumerate(self.dataset.get_corpus_iter(verbose=False)):
            doc["text"] = self.tokenize_text(doc["text"])
            yield doc

    def index(self):
        indexer = pt.IterDictIndexer(str(self.index_path))
        indexer.setProperty("tokeniser", "UTFTokeniser")
        indexer.setProperty("termpipelines", "")
        if not self.index_path.joinpath("shards.pkl").exists():
            jance_indexer = PyTDenseIndexer(self.index_path, verbose=False)
            index_path = jance_indexer.index(self.train_doc_generate())
            return index_path
        return str(self.index_path)

    def eval_on_dev(self):
        index_path = self.index()
        index_path = Path(index_path)
        if not index_path.exists():
            index_path.mkdir(parents=True)

        anceretr = PyTDenseRetrieval(index_path)

        return pt.Experiment(
            [anceretr],
            self.tokenize_topics(self.dataset),
            self.dataset.get_qrels(),
            eval_metrics=[nDCG],
            names=[self.run_name],
            save_dir=self.run_path,
            save_mode="overwrite",
        )


def main():
    dataset_pt = pt.get_dataset("irds:ntcir-transfer/1/train")
    sudachi_tokenizer = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.A

    evaluator = Evaluator(
        dataset_pt,
        sudachi_tokenizer,
        Path("../indexes/ntcir17-transfer/debug/jance"),
        "../runs/ntcir17-transfer/jance",
        tokenizer_mode=mode,
    )
    eval_result = evaluator.eval_on_dev()
    print(eval_result)


if __name__ == "__main__":
    main()
