import logging
from typing import Dict, List, Any

# from allennlp.modules.similarity_functions import BilinearSimilarity
from overrides import overrides

import torch
import numpy as np
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.common import Params
from allennlp.modules import TextFieldEmbedder, FeedForward, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, LstmSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder, BagOfEmbeddingsEncoder, BertPooler
from allennlp.nn import RegularizerApplicator, InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric
from allennlp.training.util import get_batch_size

from torch import nn
from torch.nn import Dropout, PairwiseDistance, CosineSimilarity
import torch.nn.functional as F
from torch.autograd import Variable

import warnings
import json
from copy import deepcopy

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def pad_sequence2len(tensor, dim, max_len) -> torch.LongTensor:
    shape = tensor.size()
    # print(shape)
    if shape[dim] >= max_len:
        return tensor
    
    pad_shape = list(shape)
    pad_shape[dim] = max_len - shape[dim]
    pad_tensor = torch.zeros(*pad_shape, device=tensor.device, dtype=tensor.dtype)
    new_tensor = torch.cat([tensor, pad_tensor], dim)
    return new_tensor


@Model.register("model_cnn")
class ModelCNN(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.1,
                 label_namespace: str = "class_labels",
                 device: str = "cpu",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab)

        self._device = torch.device(device)
        self._label_namespace = label_namespace
        self._dropout = Dropout(dropout)
        
        self._idx2token_label = vocab.get_index_to_token_vocabulary(namespace=label_namespace)
        self._idx_pos = vocab.get_token_index(token="pos", namespace=label_namespace)
        self._text_field_embedder = text_field_embedder
        
        # seq2vec module
        embedding_dim = self._text_field_embedder.get_output_dim()
        self._text_cnn = CnnEncoder(embedding_dim, 256, ngram_filter_sizes=(2, 3, 4, 5))
        
        text_embedding_dim = self._text_cnn.get_output_dim()

        self._num_class = self.vocab.get_vocab_size(self._label_namespace)

        self._projector = nn.Sequential(
            FeedForward(text_embedding_dim, 1, [512], torch.nn.ReLU(), dropout),  # text_header
            nn.Linear(512, self._num_class, bias=False),  # classification layer
        )

        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1-score_overall": FBetaMeasure(beta=1.0, average="weighted", labels=range(self._num_class)),  # return float
            "f1-score_each": FBetaMeasure(beta=1.0, average=None, labels=range(self._num_class))  # return list[float]
        }
        
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                sample: TextFieldTensors,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:

        output_dict = dict()
        if metadata:
            output_dict["meta"] = metadata

        # pad sequence length to 5（CNN filter size）
        sample["tokens"]["tokens"] = pad_sequence2len(tensor=sample["tokens"]["tokens"], dim=-1, max_len=5)

        mask = get_text_field_mask(sample, padding_id=0)
        sample = self._text_field_embedder(sample)
        # print(sample.shape)  # token embedding + pos embedding
        
        sample = self._text_cnn(sample, mask)

        logits = self._projector(sample)

        probs = nn.functional.softmax(logits, dim=-1)
        output_dict["probs"] = probs.tolist()
        loss = self._loss(logits, label)
        output_dict['loss'] = loss

        for metric_name, metric in self._metrics.items():
            metric(predictions=probs, gold_labels=label)

        return output_dict
    
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        # return dict()
        idx = np.argmax(output_dict["probs"], axis=1)
        out2file = list()
        for i, _ in enumerate(idx):
            out2file.append({"Issue_Url": output_dict["meta"][i]["instance"]["Issue_Url"],
                             "label": output_dict["meta"][i]["instance"]["label"],
                             "predict": self._idx2token_label[_],
                             "prob": output_dict["probs"][i][self._idx_pos]})
                             
        return out2file

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # return dict()

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

        return metrics