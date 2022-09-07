import logging
from random import sample
from typing import Dict, List, Any

# from allennlp.modules.similarity_functions import BilinearSimilarity
from overrides import overrides

import torch
import numpy as np
from allennlp.data import Vocabulary, TextFieldTensors, instance
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, LstmSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder, BagOfEmbeddingsEncoder, BertPooler
from allennlp.nn import RegularizerApplicator, InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric, metric
from allennlp.training.metrics.auc import Auc
from allennlp.training.util import get_batch_size
from torch._C import device
from .custom_metric import SiameseMeasureV1

from torch import nn
from torch.nn import Dropout, PairwiseDistance, CosineSimilarity
import torch.nn.functional as F
from torch.autograd import Variable

import warnings
import json
from copy import deepcopy

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("model_memory")
class ModelMemory(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 PTM: str = 'bert-base-uncased',
                 dropout: float = 0.1,
                 label_namespace: str = "labels",
                 device: str = "cpu",
                 use_header: bool = True,
                 temperature: float = 1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        
        super().__init__(vocab, regularizer) 
        
        self.device = torch.device(device)
        self._use_header = use_header  # whether to use the projection header

        self._label_namespace = label_namespace
        self._idx2token_label = self.vocab.get_index_to_token_vocabulary(namespace=label_namespace)
        self._dropout = Dropout(dropout)
        self._same_idx = vocab.get_token_index("same", namespace=label_namespace)
        
        self._text_field_embedder = text_field_embedder
        self._bert_pooler = BertPooler(PTM, requires_grad=True, dropout=dropout)  # pretrained tanh activation 
        
        embedding_dim = self._text_field_embedder.get_output_dim()
    
        self._num_class = self.vocab.get_vocab_size(self._label_namespace)
        if use_header:
            self._projector_single = FeedForward(embedding_dim, 1, [512], torch.nn.ReLU(), dropout)
            embedding_dim = 512
        
        self._projector = nn.Linear(3*embedding_dim, 2, bias=False)
        self._temperature = temperature

        self._golden_instances_embeddings = None
        self._golden_instances_labels = None  # list
        # self._golden_instances_ids = None  # dialogue id

        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1-score_overall": FBetaMeasure(beta=1.0, average="weighted", labels=range(self._num_class)),  # return float
            "f1-score_each": FBetaMeasure(beta=1.0, average=None, labels=range(self._num_class))  # return list[float]
        }
        self._siamese_metric = SiameseMeasureV1(self._same_idx)

        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def _instance_forward(self,
                          sample: TextFieldTensors,
                          use_header: bool = False):

        assert not use_header or hasattr(self, "_projector_single")
        sample = self._text_field_embedder(sample)
        # print(sample.shape)

        # use the CLS embedding
        sample = self._bert_pooler(sample)

        if use_header:
            sample = self._projector_single(sample)
        return sample

    def forward_gold_instances(self, sample, metadata):
        # used to get the feature vectors for anchors in the external memory
        # these feature vectors are first prepared and used during the entire inference
        embedding = self._instance_forward(sample, use_header=self._use_header)
        if not torch.is_tensor(self._golden_instances_embeddings):
            self._golden_instances_embeddings = embedding
            self._golden_instances_labels = [_["instance"][0]["label"] for _ in metadata]
                
        else:
            self._golden_instances_embeddings = torch.cat([self._golden_instances_embeddings, embedding])
            self._golden_instances_labels.extend([_["instance"][0]["label"] for _ in metadata])


    def forward(self,
                sample1: TextFieldTensors = None,
                sample2: TextFieldTensors = None,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        output_dict = dict()

        if metadata and metadata[0]["type"] == "golden":
            self.forward_gold_instances(sample1, metadata)
            return output_dict
        
        if metadata:
            output_dict["meta"] = metadata

        embedding_1 = self._instance_forward(sample1, use_header=self._use_header)
        if metadata and metadata[0]["type"] in ["test", "unlabel"]:
            shape = embedding_1.shape
            sample_embedding = embedding_1.view(shape[0], -1, shape[1])
            sample_embedding = sample_embedding.expand(-1, len(self._golden_instances_labels), -1)

            golden_embedding = self._golden_instances_embeddings.expand(shape[0], -1, -1)

            logits = self._projector(torch.cat([sample_embedding, golden_embedding, torch.abs(sample_embedding - golden_embedding)], -1))
            p = nn.functional.softmax(logits, dim=-1)  # batch_size x golden_num x 2
            output_dict['probs'] = p.tolist()
            idx_max = torch.argmax(p, dim=1)  # batch_size x 2
            idx_max = idx_max[:, self._same_idx]
            probs = [p[i][idx_max[i]] for i in range(shape[0])]  # batch_size x 2
            probs = torch.stack(probs)  # batch_size x 2

        else:
            embedding_2 = self._instance_forward(sample2, use_header=self._use_header)
            logits = self._projector(torch.cat([embedding_1, embedding_2, torch.abs(embedding_1 - embedding_2)], -1))  # concat

            probs = nn.functional.softmax(logits, dim=-1)
            # output_dict["logits"] = logits
            # output_dict["probs"] = probs.tolist()
        
            # label is not one-hot，transferred in CrossEntropyLoss
            loss = self._loss(logits / self._temperature, label)  # temperature parameter

            output_dict['loss'] = loss

        for metric_name, metric in self._metrics.items():
            metric(predictions=probs, gold_labels=label)
        
        if metadata[0]["type"] in ["test", "unlabel"]:
            self._siamese_metric(probs, metadata)  # custom metric
        return output_dict

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "meta" not in output_dict or output_dict["meta"][0]["type"] not in ["test", "unlabel"]:
            return output_dict

        out2file = list()
        vote_num = {}
        output_dict['predict'] = []
        for class_ in set(self._golden_instances_labels):
            # match score with each golden anchor
            vote_num[class_] = 0

        idx_same = self.vocab.get_token_index("same", namespace=self._label_namespace)
        for probs in output_dict['probs']:
            for p, golden_name in zip(probs, self._golden_instances_labels):
                vote_num[golden_name] = p[idx_same]
            output_dict['predict'].append(deepcopy(vote_num))
        
        for i, meta in enumerate(output_dict["meta"]):
            out2file.append({"Issue_Url": meta["instance"][0]["Issue_Url"],
                             "label": meta["instance"][0]["label"],
                             "predict": output_dict["predict"][i]})
        
        return out2file


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict()
        metrics['accuracy'] = self._metrics['accuracy'].get_metric(reset)
        precision, recall, fscore = self._metrics['f1-score_overall'].get_metric(reset).values()
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1-score'] = fscore
        precision, recall, fscore = self._metrics['f1-score_each'].get_metric(reset).values()
        for i in range(self._num_class):
            metrics[f'{self._idx2token_label[i]}_precision'] = precision[i]
            metrics[f'{self._idx2token_label[i]}_recall'] = recall[i]
            metrics[f'{self._idx2token_label[i]}_f1-score'] = fscore[i]
        
        if reset:
            # only calculate this metric when the entire evaluation is done
            siamese_metric = self._siamese_metric.get_metric(reset)
            metrics["s_precision"] = siamese_metric["precision"]
            metrics["s_recall"] = siamese_metric["recall"]
            metrics["s_f1-score"] = siamese_metric["f1"]
            metrics["s_thres"] = siamese_metric["thres"]
            metrics["s_auc"] = siamese_metric["auc"]
            metrics["s_ave_precision_score"] = siamese_metric["ave_precision_score"]

        return metrics
    

    def get_output_dim(self, use_header=False):
        assert not use_header or hasattr(self, "_projector_single")
        if use_header:
            return self._projector_single.get_output_dim()
        return self._text_field_embedder.get_output_dim()