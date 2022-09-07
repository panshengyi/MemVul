import torch
import allennlp
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.data_loaders.data_loader import DataLoader
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from typing import Dict, Any

from overrides import overrides
from .reader_memory import ReaderMemory
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@TrainerCallback.register("reset_dataloader")
class ResetLoader(TrainerCallback):
    # for online negative sampling, training set for each epoch is different
    def __init__(self, serialization_dir: str = None) -> None:
        super().__init__(serialization_dir)
        
    @overrides
    def on_epoch(self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int, is_primary: bool, **kwargs) -> None:
        # data_loader will re-read the dataset before next epoch (all the pos samples and re-sampled negative samples)
        trainer.data_loader._instances = None


@TrainerCallback.register("custom_validation")
class CustomValidation(TrainerCallback):
    def __init__(self,
                 anchor_path: str,
                 data_reader: DatasetReader = None,
                 data_loader: DataLoader = None,
                 serialization_dir: str = None) -> None:
        super().__init__(serialization_dir)
        # for custom validation, this reader is used to read the golden anchor
        PTM = "bert-base-uncased"
        reader = data_reader or ReaderMemory(tokenizer = PretrainedTransformerTokenizer(PTM, add_special_tokens=True, max_length=512),
                                          token_indexers = {"tokens": PretrainedTransformerIndexer(PTM, namespace="tags")})
        self._anchors = list(reader.read(anchor_path))
        
    @overrides
    def on_epoch(self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int, is_primary: bool, **kwargs) -> None:
        # not in a distributed mode，trainer.model euqals to trainer._pytorch_model
        model = trainer.model
        model.eval()

        model._golden_instances_embeddings = None  # reset
        model._golden_instances_labels = None  # reset
        logger.info("updating golden embeddings")
        model.forward_on_instances(self._anchors[:128])
        if len(self._anchors) > 128:
            model.forward_on_instances(self._anchors[128:])
        
        # with torch.no_grad():
        #     trainer._validation_loss(epoch)
        



        