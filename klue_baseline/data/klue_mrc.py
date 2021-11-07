import argparse
import json
import logging
import os
from typing import Any, List, Optional

from overrides import overrides
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.data.processors.squad import SquadExample, squad_convert_examples_to_features

from klue_baseline.data.base import DataProcessor, KlueDataModule

logger = logging.getLogger(__name__)


class KlueMRCExample(SquadExample):
    """A single example for KLUE-MRC of transformer.data.processor.squad.SquadExample

    Args:
        question_type: The type number of the question among 1 to 3. 1 for
            paraphrasing. 2 for multiple-sentences. 3 for unanswerable.
    """

    def __init__(self, question_type: int, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.question_type = question_type


class KlueMRCProcessor(DataProcessor):

    origin_train_file_name: str = "x_train.csv"
    origin_dev_file_name: str = "x_valid.csv"
    origin_test_file_name: str = "x_test.csv"

    datamodule_type = KlueDataModule

    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(args, tokenizer)

    @overrides
    def get_train_dataset(self, data_dir: str, file_name: Optional[str] = None) -> Any:
        file_path = os.path.join(data_dir, file_name or self.origin_train_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "train")

    @overrides
    def get_dev_dataset(self, data_dir: str, file_name: Optional[str] = None) -> Any:
        file_path = os.path.join(data_dir, file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "valid")

    @overrides
    def get_test_dataset(self, data_dir: str, file_name: Optional[str] = None) -> Any:
        file_path = os.path.join(data_dir, file_name or self.origin_test_file_name)

        if not os.path.exists(file_path):
            logger.info("Test dataset doesn't exists. So loading dev dataset instead.")
            file_path = os.path.join(data_dir, self.hparams.dev_file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "test")

    def _create_dataset(self, file_path: str, dataset_type: str) -> Any:
        is_training = dataset_type == "train"
        examples = self._create_examples(file_path, is_training=is_training)
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.hparams.max_seq_length,
            doc_stride=self.hparams.doc_stride,
            max_query_length=self.hparams.max_query_length,
            is_training=is_training,
            return_dataset="pt",
            threads=10,
        )

        if not is_training:
            data = getattr(self.hparams, "data", {})
            data[dataset_type] = {"examples": examples, "features": features}
            setattr(self.hparams, "data", data)

        return dataset

    def _create_examples(self, file_path: str, is_training: bool = True, data_split_ratio: int = 1) -> List[KlueMRCExample]:
        input_data = pd.read_csv(file_path)
        start_pos, end_pos = 0, len(input_data)
        if data_split_ratio != 1:
            split_ratio = (data_split_ratio % 100) / 100
            chunk = int(len(input_data) * split_ratio)
            start_pos = int(chunk * (data_split_ratio // 100))
            end_pos = int(start_pos + chunk)
            
        examples = []
        for i, entry in tqdm(enumerate(input_data[start_pos:end_pos].itertuples())):
            # print(i, entry)
            example = KlueMRCExample(
                question_type=1,
                qas_id="aihub-mrc-v1_train_"+format(i, '06'),
                question_text=entry.question,
                context_text=entry.text,
                answer_text=entry.answer,
                start_position_character=entry.answer_start,
                title=entry.title,
                is_impossible=False,
            )
            examples.append(example)

        return examples

    def _create_examples_backup(self, file_path: str, is_training: bool = True) -> List[KlueMRCExample]:
        with open(file_path, "r", encoding="utf-8") as reader:
            input_data = json.load(reader)["data"]

        examples = []
        for entry in tqdm(input_data):
            for paragraph in entry["paragraphs"]:
                for qa in paragraph["qas"]:
                    is_impossible = qa.get("is_impossible", False)
                    if not is_training:
                        answers = qa["answers"]
                        answer_text = None
                        start_position_character = None
                    elif is_training and not is_impossible:
                        answers = []
                        answer_text = qa["answers"][0]["text"]
                        start_position_character = qa["answers"][0]["answer_start"]
                    elif is_training and is_impossible:
                        answers = []
                        answer_text = ""
                        start_position_character = -1

                    example = KlueMRCExample(
                        question_type=qa.get("question_type", 1),
                        qas_id=qa["guid"],
                        question_text=qa["question"],
                        context_text=paragraph["context"],
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=entry["title"],
                        answers=answers,
                        is_impossible=is_impossible,
                    )
                    examples.append(example)
        return examples

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        parser = KlueDataModule.add_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            default=510,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded.",
        )
        parser.add_argument(
            "--doc_stride",
            default=128,
            type=int,
            help="When splitting up a long document into chunks, how much stride to take between chunks.",
        )
        parser.add_argument(
            "--max_query_length",
            default=64,
            type=int,
            help="The maximum number of tokens for the question. Questions longer than this will "
            "be truncated to this length.",
        )
        return parser
