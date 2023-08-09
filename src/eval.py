import logging
import os
import re
from importlib import reload
from pathlib import Path

import pyterrier as pt
from pyterrier.datasets import Dataset
from pyterrier.measures import MRR, R, nDCG
from pyterrier.transformer import TransformerBase
from sudachipy import dictionary, tokenizer
import pandas as pd

import ntcir_datasets.ntcir_transfer
from models.loader import LoadRetriever
from utils import is_debug, project_dir

if not pt.started():
    pt.init()


class Evaluator(object):
    def __init__(
        self,
        dataset: Dataset,
        sudachi_tokenizer: tokenizer.Tokenizer,
        indexer: TransformerBase,
        retriever: TransformerBase,
        run_path: Path,
        run_name: str = "kasys",
        tokenizer_mode: str = tokenizer.Tokenizer.SplitMode.A,
    ):
        self.dataset = dataset
        self.sudachi_tokenizer = sudachi_tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.indexer = indexer
        self.retriever = retriever

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
            if is_debug() and i < 100:
                break

    def index(self) -> str:
        index_path = self.indexer.index(self.train_doc_generate())
        return str(index_path)

    def eval_on_dev(self):
        self.index()

        return pt.Experiment(
            [self.retriever],
            self.tokenize_topics(self.dataset),
            self.dataset.get_qrels(),
            eval_metrics=[nDCG, nDCG@10, nDCG@100, nDCG @ 1000, R @ 100, ],
            names=[self.run_name],
            save_dir=str(self.run_path),
            save_mode="overwrite",
        )

    def eval_on_eval(self):
        def gen_dummy_qrels():
            dummy_qrels = pd.DataFrame(self.dataset.get_topics(), columns=['qid'])
            dummy_qrels['docno'] = 'docno'
            dummy_qrels['label'] = 0
            return dummy_qrels

        self.index()

        return pt.Experiment(
            [self.retriever],
            self.tokenize_topics(self.dataset),
            gen_dummy_qrels(),
            eval_metrics=[nDCG, nDCG@10, nDCG@100, nDCG @ 1000, R @ 100, ],
            names=[self.run_name],
            save_dir=str(self.run_path),
            save_mode="overwrite",
        )


def main():
    dataset_name = "ntcir-transfer/1/dev"
    # model_name = "contriever-msmarco"
    # model_name = "contriever-mrtidy"
    model_name = "contriever-transfer"
    # model_name = "tevatron-contriever-mrtidy"

    dataset_pt = pt.get_dataset(f"irds:{dataset_name}")
    sudachi_tokenizer = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.A

    indexer, retriever = LoadRetriever(dataset_name, model_name).load_retriever()

    run_path = project_dir.joinpath("runs/ntcir17-transfer/dev", model_name)
    if not run_path.exists():
        run_path.mkdir(parents=True)

    evaluator = Evaluator(
        dataset_pt,
        sudachi_tokenizer,
        indexer,
        retriever,
        run_path,
        run_name=model_name,
        tokenizer_mode=mode,
    )
    eval_result = evaluator.eval_on_dev()
    print(eval_result)

def gen_runs():
    dataset_name = "ntcir-transfer/1/eval"
    model_name = "contriever-msmarco"
    # model_name = "contriever-mrtidy"
    # model_name = "contriever-transfer"
    # model_name = "tevatron-contriever-mrtidy"

    dataset_pt = pt.get_dataset(f"irds:{dataset_name}")
    sudachi_tokenizer = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.A

    indexer, retriever = LoadRetriever(dataset_name, model_name).load_retriever()

    run_path = project_dir.joinpath("runs/ntcir17-transfer/eval", model_name)
    if not run_path.exists():
        run_path.mkdir(parents=True)

    evaluator = Evaluator(
        dataset_pt,
        sudachi_tokenizer,
        indexer,
        retriever,
        run_path,
        run_name=model_name,
        tokenizer_mode=mode,
    )
    eval_result = evaluator.eval_on_eval()
    print(eval_result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # main()
    gen_runs()
