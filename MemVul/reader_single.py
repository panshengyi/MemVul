import json
import random
import re
from allennlp import data
import numpy as np
from collections import defaultdict
from itertools import permutations
from typing import Dict, List, Optional
import logging
from datetime import datetime
from copy import deepcopy

from allennlp.data import Field
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer

from .util import replace_tokens_simple

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("reader_single")
class ReaderSingle(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 sample_neg: float = None,
                 train_iter: int = None,
                 cache_directory: Optional[str] = None,
                 target: str = "Security_Issue_Full") -> None:
        super().__init__()

        self._token_indexers = token_indexers  # token indexers for text
        self._tokenizer = tokenizer
        self._target = target
        # self._sample_weights = {'pos': 1, 'neg': 1}
        self._choice_neg = [True, False]
        select_neg = sample_neg or 0.1  #  ratio for sampling the negatives 
        self._train_iter = train_iter or 1
        self._select_neg = [select_neg, 1 - select_neg]  # [True, False]
        self._dataset = dict()  # key is the file path

    def read_dataset(self, file_path):
        if self._dataset.get(file_path):
            # the results of tokenization is reusable, don't need to do it again 
            return self._dataset[file_path]
        
        samples = json.load(open(file_path, 'r', encoding="utf-8"))

        dataset = dict()
        for s in samples:
            s["description"] = self._tokenizer.tokenize(f"{s['Issue_Title']}. {s['Issue_Body']}")  # merge the issue title and body as the input to the model

            label = "pos" if str(s[self._target]) == "1" else "neg"
            s[self._target] = label
            if dataset.get(label) is None:
                dataset[label] = list()
            dataset[label].append(s)
        
        self._dataset[file_path] = dataset

        return dataset

    @overrides
    def _read(self, file_path):
        dataset = self.read_dataset(file_path)
        
        all_data = []
        for ll in list(dataset.values()):
            all_data.extend(ll)

        classes_districution = dict()
        for k, v in dataset.items():
            classes_districution[k] = len(v)
        logger.info(classes_districution)

        if "test_" in file_path:
            # path may accidentally contain the keywords, hence adding the userline
            logger.info("loading unlabel examples ...")
            for sample in all_data:
                yield self.text_to_instance(sample, type_="unlabel")
            logger.info(f"Num of unlabel instances is {len(all_data)}")

        elif "validation_" in file_path:
            logger.info("loading testing examples ...")
            for sample in all_data:
                yield self.text_to_instance(sample, type_="test")
            logger.info(f"Num of testing instances is {len(all_data)}")
            
        else:
            random.shuffle(all_data)  # must shuffle for training
            # training
            logger.info("loading training examples ...")
            num_train = 0
            iter_num = self._train_iter
            
            for i in range(iter_num):
                for sample in all_data:
                    key = sample[self._target]
                    # sample the negative samples
                    if key == "pos" or random.choices(self._choice_neg, weights=self._select_neg, k=1)[0]:
                        yield self.text_to_instance(sample, type_="train")
                        num_train += 1
            logger.info(f"Num of training instances is {num_train}")

    @overrides
    def text_to_instance(self, ins, type_="train") -> Instance:  # type: ignore
        # share the code between predictor and trainer, hence the label field is optional
        fields: Dict[str, Field] = {}

        fields["sample"] = TextField(ins["description"], self._token_indexers)

        fields['label'] = LabelField(ins[self._target], label_namespace="class_labels")
        
        meta_ins = {"Issue_Url": ins["Issue_Url"], "label": ins[self._target]}
        fields['metadata'] = MetadataField({"type": type_, "instance": meta_ins})

        return Instance(fields)