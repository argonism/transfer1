from pathlib import Path

from pyterrier.transformer import TransformerBase
from utils import index_base_path

from .jance.jance import PyTDenseIndexer, PyTDenseRetrieval
from .pt_contriever import ContrieverIndexer, ContrieverRetrieval
from .pt_tevatron import TevatronIndexer, TevatronRetrieval

LoadRetrieverReturn = tuple[TransformerBase, TransformerBase]


class LoadRetriever:
    def __init__(self, dataset_name: str, model_name: str) -> None:
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.index_path = self.__index_path()
        self.batch_size = 16

    def __index_path(self):
        return index_base_path.joinpath(self.dataset_name, self.model_name)

    def load_jance(self) -> LoadRetrieverReturn:
        return (
            PyTDenseIndexer(self.index_path, verbose=False),
            PyTDenseRetrieval(self.index_path),
        )

    def load_contriever(self) -> LoadRetrieverReturn:
        return (
            ContrieverIndexer(self.index_path, verbose=False),
            ContrieverRetrieval(self.index_path),
        )

    def load_contriever_msmarco(self) -> LoadRetrieverReturn:
        return (
            ContrieverIndexer(
                self.index_path, model_path="facebook/mcontriever-msmarco", verbose=False
            ),
            ContrieverRetrieval(
                self.index_path,
                model_path="facebook/mcontriever-msmarco",
            ),
        )

    def load_contriever_mrtidy(self) -> LoadRetrieverReturn:
        model_path = Path(__file__).parent.joinpath(
            # "contriever/model/mrtidy_jp/pretrained/step-3000"
            # "contriever/model_mrtydi_japanese/checkpoint-100"
            "contriever/model_ntcir_train/checkpoint-10"
        )
        return (
            ContrieverIndexer(
                self.index_path, model_path=model_path, verbose=False, overwrite=True
            ),
            ContrieverRetrieval(
                self.index_path,
                model_path=model_path,
            ),
        )

    def load_contriever_transfer(self) -> LoadRetrieverReturn:
        model_path = Path(__file__).parent.joinpath(
            "contriever/model/transfer/pretrained/step-1000"
        )
        return (
            ContrieverIndexer(self.index_path, model_path=model_path, verbose=False),
            ContrieverRetrieval(
                self.index_path,
                model_path=model_path,
            ),
        )

    def load_tevatron_contriever_mrtidy(self) -> LoadRetrieverReturn:
        model_path = Path(__file__).parent.joinpath(
            "contriever/model_mrtydi_japanese/checkpoint-5"
        )
        return (
            TevatronIndexer(self.index_path, model_path=model_path, verbose=False),
            TevatronRetrieval(
                self.index_path,
                model_path=model_path,
            ),
        )

    def load_retriever(self) -> LoadRetrieverReturn:
        if self.model_name == "jance":
            return self.load_jance()
        elif self.model_name == "contriever":
            return self.load_contriever()
        elif self.model_name == "contriever-msmarco":
            return self.load_contriever_msmarco()
        elif self.model_name == "contriever-mrtidy":
            return self.load_contriever_mrtidy()
        elif self.model_name == "contriever-transfer":
            return self.load_contriever_transfer()
        elif self.model_name == "tevatron-contriever-mrtidy":
            return self.load_tevatron_contriever_mrtidy()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
