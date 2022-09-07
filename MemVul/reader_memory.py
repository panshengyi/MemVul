from dataclasses import field, fields
from enum import Flag
import json
from json import encoder
import random
import re
from allennlp import data
from allennlp.data.fields.text_field import TextFieldTensors
import numpy as np
from collections import defaultdict
from itertools import permutations
from typing import Dict, List, Optional, Text
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
from transformers.utils.dummy_pt_objects import ElectraForMaskedLM

from .util import replace_tokens_simple

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("reader_memory")
class ReaderMemory(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 same_diff_ratio: Dict[str, int] = None,
                 target: str = "Security_Issue_Full",
                 anchor_path: str = "CWE_anchor_golden_project.json",
                 sample_neg: float = None,
                 train_iter: int = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        # super().__init__(cache_directory=cache_directory)
        super().__init__()

        self._token_indexers = token_indexers
        self._tokenizer = tokenizer
        self._same_diff_ratio = same_diff_ratio or {"diff": 6, "same": 2}
        self._choice_neg = [True, False]
        select_neg = sample_neg or 0.1
        self._train_iter = train_iter or 1
        self._select_neg = [select_neg, 1 - select_neg]
        self._target = target
        
        if sample_neg is None:
            # when used in the callbacks for custom validation (loading golden anchors)
            return
        
        data_path = "xxx"
        # get the CWE ID from the correspoding CVE record
        self._cve_info = json.load(open(data_path + 'CVE_dict.json', 'r'))  # dict
        # get the anchors
        self._anchor = json.load(open(anchor_path, 'r'))  # used for constructing pairs during training
        for k, v in self._anchor.items():
            self._anchor[k] = self._tokenizer.tokenize(v)
        
        self._dataset = dict()

    def read_dataset(self, file_path):
        if "golden" in file_path:
            # for anchors in the external memory
            dataset = dict()
            anchors = json.load(open(file_path, 'r', encoding="utf-8"))  # dict
            for cwe_id, description in anchors.items():
                dataset[cwe_id] = [{self._target: cwe_id, "description": self._tokenizer.tokenize(description)}]
            return dataset
            
        if self._dataset.get(file_path):
            return self._dataset[file_path]
            
        samples = json.load(open(file_path, 'r', encoding="utf-8"))

        dataset = {"neg": list()}  # for pos data, we use the CWE ID for key
        for s in samples:
            s["description"] = self._tokenizer.tokenize(f"{s['Issue_Title']}. {s['Issue_Body']}")

            # "1" for pos and "0" for neg
            label = "pos" if str(s[self._target]) == "1" else "neg"
            s[self._target] = label
            if label == "pos":
                # pos sample
                cve_id = s["CVE_ID"]
                if type(self._cve_info[cve_id]["CVE_Description"]) == str:
                    # need to perform special token replacement for CVE
                    self._cve_info[cve_id]["CVE_Description"] = replace_tokens_simple(self._cve_info[cve_id]["CVE_Description"])
                    self._cve_info[cve_id]["CVE_Description"] = self._tokenizer.tokenize(self._cve_info[cve_id]["CVE_Description"])
                
                s["CWE_ID"] = self._cve_info[cve_id]["CWE_ID"]
                label = s["CWE_ID"]
                if label is None:
                    # 2 dirty data
                    continue
                if label not in dataset:
                    dataset[label] = list()
                
            dataset[label].append(s)
        
        self._dataset[file_path] = dataset  # update

        return dataset

    @overrides
    def _read(self, file_path):
        dataset = self.read_dataset(file_path)
        all_data = list()
        min_sample_num = 9999
        for ll in list(dataset.values()):
            if len(ll) < min_sample_num:
                min_sample_num = len(ll)
            all_data.extend(ll)

        classes_districution = {'pos': 0, 'neg': None}
        pos_classes = list()
        for k, v in dataset.items():
            if k != 'neg':
                classes_districution['pos'] += len(v)
                pos_classes.append(k)
            else:
                classes_districution['neg'] = len(v)
        logger.info(classes_districution)

        same_num = 0
        diff_num = 0

        if "golden_" in file_path:
            # path may accidentally contain the keywords, hence adding the userline
            # provide golden instances
            logger.info("Begin loading golden instances------")
            for sample in all_data:
                yield self.text_to_instance((sample, sample), type_="golden")
            logger.info(f"Num of golden instances is {len(all_data)}")

        elif "test_" in file_path:
            # provide test data
            logger.info("Begin predict------")

            for sample in reversed(all_data):
                # positives come first and then the negatives 
                yield self.text_to_instance((sample, sample), type_="unlabel")
            logger.info(f"Predict sample num is {len(all_data)}")

        elif "validation_" in file_path:
            # provide valdiation data
            logger.info("Begin testing------")
            num_test_sample = 0
            for sample in reversed(all_data):
                num_test_sample += 1
                yield self.text_to_instance((sample, sample), type_="test")
            logger.info(f"Test sample num is {num_test_sample}")

        else:
            # must shuffle for train
            random.shuffle(all_data)

            iter_num = self._train_iter
            anchor_classes = list(self._anchor.keys())

            same_per_sample = self._same_diff_ratio["same"]  # number of matched pairs (CIR)
            diff_per_sample = self._same_diff_ratio["diff"]  # number of mismatched pairs (NCIR)

            for _ in range(iter_num):
                for sample in all_data:
                    key = sample[self._target]
                    if key == "pos":
                        
                        yield self.text_to_instance((sample, sample), type_="train")  # always use the corresponding CVE to make pairs
                        for same in random.choices(dataset[sample['CWE_ID']], k = same_per_sample - 1):
                            yield self.text_to_instance((sample, same), type_="train")  # CVE pair

                        same_num += same_per_sample  # matched pairs
                    
                    elif random.choices(self._choice_neg, weights=self._select_neg, k=1)[0]:
                        for diff in random.choices(anchor_classes, k = diff_per_sample):
                            # random sample k anchors from the external memory
                            yield self.text_to_instance((sample, {"CWE_ID": diff, self._target: "pos"}), type_="train")
                        
                        diff_num += diff_per_sample
            
            logger.info(f"Dataset Count: Same : {same_num} / Diff : {diff_num}")

    @overrides
    def text_to_instance(self, p, type_="train") -> Instance:
        fields: Dict[str, Field] = dict()
        ins1, ins2 = p # instance:Dict{id, intention, messages} mess:Dict{id, text, time, index, user}

        fields["sample1"] = TextField(ins1["description"], self._token_indexers)
        ins1_class = ins1[self._target]
        ins2_class = ins2[self._target]
        
        if type_ == "train":
            # always true
            if ins2_class == "pos":
                if ins1_class == "neg":
                    # use description of the anchor
                    fields["sample2"] = TextField(self._anchor[ins2["CWE_ID"]], self._token_indexers)
                elif ins1["Issue_Url"] == ins2["Issue_Url"]:
                    # use description of the corresponding CVE
                    fields["sample2"] = TextField(self._cve_info[ins2["CVE_ID"]]["CVE_Description"], self._token_indexers)
                elif random.choices([True, False], [0.7, 0.3], k=1)[0]:
                    # use description of the other CVE that belong to the same category
                    fields["sample2"] = TextField(self._cve_info[ins2["CVE_ID"]]["CVE_Description"], self._token_indexers)
                elif random.choices([True, False], [0.5, 0.5], k=1)[0]:
                    # use description of the anchor
                    anchor_id = ins2["CWE_ID"]
                    if anchor_id is not None:
                        fields["sample2"] = TextField(self._anchor[anchor_id], self._token_indexers)
                    else:
                        fields["sample2"] = TextField(ins2["description"], self._token_indexers)
                else:
                    # use description of other issue report that belong to the same category
                    fields["sample2"] = TextField(ins2["description"], self._token_indexers)

        if type_ in ["train"]:
            if ins1[self._target] == ins2[self._target]:
                fields['label'] = LabelField("same")
            else:
                fields['label'] = LabelField("diff")
        elif type_ in ["test", "unlabel"]:
            # pos == same (we only use CIR to make matched pairs)
            # neg == diff (we only use NCIR to make mismatched pairs)
            if ins1[self._target] == "pos":
                fields['label'] = LabelField("same")
            else:
                fields['label'] = LabelField("diff")
        
        meta_ins1 = {"label": ins1[self._target]}
        if type_ in ["train", "test", "unlabel"]:
            if ins1[self._target] == "pos":
                meta_ins1["label"] = ins1["CWE_ID"]
            meta_ins1["Issue_Url"] = ins1["Issue_Url"]
        
        fields['metadata'] = MetadataField({"type": type_, "instance": [meta_ins1]})  # only to record information
        return Instance(fields)