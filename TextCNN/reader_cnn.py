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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("reader_cnn")
class ReaderCNN(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 target: str = "Security_Issue_Full",
                 sample_neg: float = None,
                 train_iter: int = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        
        super().__init__()

        self._token_indexers = token_indexers
        self._tokenizer = tokenizer
        
        self._choice_neg = [True, False]
        select_neg = sample_neg or 0.1
        self._train_iter = train_iter or 1
        self._select_neg = [select_neg, 1 - select_neg]
        self._target = target

        self._dataset = dict()

    def read_dataset(self, file_path):
        if self._dataset.get(file_path):
            # the results of tokenization is reusable, don't need to do it again 
            return self._dataset[file_path]
        
        samples = json.load(open(file_path, 'r', encoding="utf-8"))

        dataset = dict()
        for i, s in enumerate(samples):
            # we do not perform the tokenization here, because not all negatives are used in training
            s["description"] = f"{s['Issue_Title']}. {s['Issue_Body']}"

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

        if "test" in file_path:
            logger.info("loading unlabel examples ...")

            for sample in all_data:
                yield self.text_to_instance(sample, type_="unlabel")

            logger.info(f"Num of unlabel instances is {len(all_data)}")

        elif "validation" in file_path:
            logger.info("loading testing examples ...")

            for sample in all_data:
                yield self.text_to_instance(sample, type_="test")
            
            logger.info(f"Num of testing instances is {len(all_data)}")
            
        else:
            # must shuffle for train
            random.shuffle(all_data)

            logger.info("loading training examples ...")

            iter_num = self._train_iter
            num_train = 0
            
            for _ in range(iter_num):
                for sample in all_data:
                    key = sample[self._target]
                    # sample the negative samples
                    if key == "pos" or random.choices(self._choice_neg, weights=self._select_neg, k=1)[0]:
                        yield self.text_to_instance(sample, type_="train")
                        num_train += 1
            
            logger.info(f"Num of training instances is {num_train}")

    @overrides
    def text_to_instance(self, ins, type_="train") -> Instance: 
        fields: Dict[str, Field] = {}

        fields["sample"] = TextField(self._tokenizer.tokenize(ins["description"]), self._token_indexers)

        fields['label'] = LabelField(ins[self._target], label_namespace="class_labels")
        
        meta_ins = {"Issue_Url": ins["Issue_Url"], "label": ins[self._target]}
        fields['metadata'] = MetadataField({"type": type_, "instance": meta_ins})

        return Instance(fields)