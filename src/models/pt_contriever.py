from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, Union

import faiss
import more_itertools
import numpy as np
import pandas as pd
import pyterrier as pt
import torch
from pyterrier.model import add_ranks
from pyterrier.transformer import TransformerBase
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel

sys.path.append(str(Path(__file__).parent.joinpath("contriever/")))

from .contriever.src.contriever import Contriever, load_retriever

logger = getLogger(__name__)


def load_contriever(
    model_path: Union[str, Path], device: str = "cuda:0"
) -> tuple[BertModel, AutoTokenizer]:
    if isinstance(model_path, Path) and model_path.exists():
        logger.info(f"loading contriever from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mcontriever-msmarco")
        encoder = Contriever.from_pretrained(model_path)
        # retriever, tokenizer, retriever_model_id = load_retriever(str(model_path))
        # logger.info(f"retriever_model_id: {retriever_model_id}")
        # tokenizer = tokenizer
        # encoder = retriever
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        encoder = Contriever.from_pretrained(model_path)
    encoder.to(device)
    encoder.eval()
    return encoder, tokenizer


class ContrieverIndexer(TransformerBase):
    def __init__(
        self,
        index_path: Path,
        model_path: str = "facebook/mcontriever",
        num_docs: Optional[int] = None,
        verbose: bool = True,
        segment_size: int = 500_000,
        device: str = "cuda:0",
        batch_size: int = 16,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available():
            logger.warn("cuda is not avaliable. AnceEncoder use cpu as device")
            device = "cpu"
        self.index_path = index_path
        self.model_path = model_path
        self.device = device

        self.encoder, self.tokenizer = load_contriever(self.model_path)
        self.max_length = 512
        self.query_max_length = 64
        self.verbose = verbose
        self.num_docs = num_docs
        if self.verbose and self.num_docs is None:
            raise ValueError("if verbose=True, num_docs must be set")
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.overwrite = overwrite

    def index(self, generator):
        if not self.overwrite and self.index_path.joinpath("shards.pkl").exists():
            return self.index_path

        os.makedirs(self.index_path, exist_ok=True)

        docid2docno = []

        def gen_tokenize():
            kwargs = {}
            if self.num_docs is not None:
                kwargs["total"] = self.num_docs
            for doc in (
                pt.tqdm(generator, desc="Indexing", unit="d", **kwargs)
                if self.verbose
                else generator
            ):
                docid2docno.append(doc["docno"])

                yield doc["text"]

        segment = -1
        shard_size = []
        for docs in more_itertools.ichunked(gen_tokenize(), self.segment_size):
            segment += 1

            print("Segment %d" % segment)
            docs = list(docs)
            passage_embedding = torch.tensor([])
            for batch_offset in tqdm(range(0, len(docs), self.batch_size)):
                batch_docs = docs[batch_offset : batch_offset + self.batch_size]
                inputs = self.tokenizer(
                    batch_docs, padding=True, truncation=True, return_tensors="pt"
                )
                inputs.to(self.device)
                embeddings = self.encoder(**inputs)
                batch_passage_embedding = embeddings.detach().cpu()
                passage_embedding = torch.cat(
                    (passage_embedding, batch_passage_embedding), dim=0
                )

            shard_file_path = self.index_path.joinpath(str(segment) + ".pkl")
            shard_file_path.write_bytes(pickle.dumps(passage_embedding))

            passage_embedding = None

            shard_size.append(len(docs))

        with pt.io.autoopen(os.path.join(self.index_path, "shards.pkl"), "wb") as f:
            pickle.dump(shard_size, f)
            pickle.dump(docid2docno, f)
        return self.index_path


class ContrieverRetrieval(TransformerBase):
    def __init__(
        self,
        index_path: Path,
        model_path: str = "facebook/mcontriever",
        num_results=10000,
        device: str = "cuda:0",
        **kwargs,
    ):
        self.num_results = num_results
        self.model_path = model_path
        self.device = device

        self.encoder, self.tokenizer = load_contriever(self.model_path)

        self.max_length = 512
        self.query_max_length = 64
        self.index_path = index_path

    def load_shard_metadata(self) -> None:
        logger.debug("Loading shard metadata")
        shards_files = os.path.join(self.index_path, "shards.pkl")
        with pt.io.autoopen(shards_files) as f:
            self.shard_sizes = pickle.load(f)
            self.docid2docno = pickle.load(f)
        self.segments = len(self.shard_sizes)

    def yield_shard_indexes(
        self, shard_sizes: list[int], index_path: Path
    ) -> Generator[tuple[Any, int], None, None]:
        offset = 0
        for i, shard_size in enumerate(
            tqdm(shard_sizes, desc="Loading shards", unit="shard")
        ):
            shard_path = index_path.joinpath(str(i) + ".pkl")
            passage_embs = pickle.loads(shard_path.read_bytes())

            yield passage_embs, offset

            offset += shard_size

    def __str__(self) -> str:
        return "PyTDenseIndexer"

    def calc_scores_with_faiss(
        self, passage_embs: np.ndarray, query_embs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dim = passage_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(passage_embs)

        faiss.omp_set_num_threads(16)
        scores, neighbours = index.search(query_embs, self.num_results)
        return scores, neighbours

    def calc_scores_naive(self, passage_embs: np.ndarray, query_embs: np.ndarray):
        scores = np.matmul(query_embs, passage_embs.T)
        sorted_scores = []
        neighbours = []
        for score_list in scores:
            sorted_i_score = sorted(
                enumerate(score_list), key=lambda x: x[1], reverse=True
            )
            sorted_scores.append([score for _, score in sorted_i_score])
            neighbours.append([i for i, _ in sorted_i_score])
        return np.array(sorted_scores), np.array(neighbours)

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        self.load_shard_metadata()

        queries = topics["query"].to_list()
        qid2q = {qid: q for q, qid in zip(queries, topics["qid"].to_list())}

        print("***** inference of %d queries *****" % len(qid2q))
        inputs = self.tokenizer(
            queries, padding=True, truncation=True, return_tensors="pt"
        )
        inputs.to(self.device)
        embeddings = self.encoder(**inputs)
        query_embeddings = embeddings.detach().cpu().numpy()

        print(
            "***** faiss search for %d queries on %d shards *****"
            % (len(qid2q), self.segments)
        )
        rtr = []
        indexes_iter = self.yield_shard_indexes(self.shard_sizes, self.index_path)
        for passage_embs, offset in tqdm(indexes_iter, desc="Calc Scores"):
            if isinstance(passage_embs, torch.Tensor):
                passage_embs = passage_embs.numpy()
            scores, neighbours = self.calc_scores_with_faiss(
                passage_embs, query_embeddings
            )

            res = self._calc_scores(
                topics["qid"].values,
                neighbours,
                scores,
                qid2q,
                num_results=self.num_results,
                offset=offset,
            )
            rtr.append(res)
        rtr = pd.concat(rtr)
        rtr = add_ranks(rtr)
        rtr = rtr[rtr["rank"] < self.num_results]
        rtr = rtr.sort_values(
            by=["qid", "score", "docno"], ascending=[True, False, True]
        )
        return rtr

    def _calc_scores(
        self,
        query_embedding2id: list[str],
        I_nearest_neighbor: np.ndarray,
        I_scores: np.ndarray,
        qid2q: dict[str, str],
        num_results: int = 10000,
        offset: int = 0,
    ) -> pd.DataFrame:
        rtr = []
        for query_idx in range(I_nearest_neighbor.shape[0]):
            query_id = query_embedding2id[query_idx]

            top_ann_pid = I_nearest_neighbor[query_idx, :].copy()
            scores = I_scores[query_idx, :].copy()
            selected_ann_idx = top_ann_pid[:num_results]
            rank = 0
            seen_pid = set()

            for i, idx in enumerate(selected_ann_idx):
                rank += 1
                docno = self.docid2docno[idx + offset]
                rtr.append([query_id, qid2q[query_id], idx, docno, rank, scores[i]])
                seen_pid.add(idx)

        return pd.DataFrame(
            rtr, columns=["qid", "query", "docid", "docno", "rank", "score"]
        )
