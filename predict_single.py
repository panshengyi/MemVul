import re
from allennlp.training.metrics import metric
# import torch
from allennlp.data import Instance, Token, Vocabulary, allennlp_collate
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder

from allennlp.common import Params
from allennlp.models import Model
import json
import importlib
import numpy as np
from typing import Dict, Any
import random

import argparse
import json
import logging
from typing import Any, Dict

# from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common.logging import prepare_global_logging
from allennlp.common.util import prepare_environment, import_module_and_submodules
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate
from allennlp.common.tqdm import Tqdm
import matplotlib.pyplot as plt
import pickle as pkl

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from predict_memory import model_measure

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)


def test(archive_file, input_file, test_config=None, weights_file=None, output_file=None, predictions_output_file=None, batch_size=64, cuda_device=0, seed=2021, package="MemVul", batch_weight_key="", file_friendly_logging=False) -> Dict[str, Any]:
    
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    import_module_and_submodules(package)
    overrides = test_config or ""

    archive = load_archive(
        archive_file,
        weights_file=weights_file,
        cuda_device=cuda_device,
        overrides=overrides,
    )
    config = archive.config
    prepare_environment(config)
    model = archive.model

    model.eval()
    
    dataset_reader = archive.dataset_reader  # load test samples

    evaluation_data_path = input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)

    data_loader_params = config.pop("validation_data_loader", None)
    if data_loader_params is None:
        data_loader_params = config.pop("data_loader")
    if batch_size:
        data_loader_params["batch_size"] = batch_size
    data_loader = DataLoader.from_params(
        params=data_loader_params, reader=dataset_reader, data_path=evaluation_data_path
    )

    data_loader.index_with(model.vocab)

    metrics = evaluate(
        model,
        data_loader,
        cuda_device,
        batch_weight_key,
        output_file=output_file,
        predictions_output_file=predictions_output_file,
    )

    logger.info("Finished evaluating.")

    return metrics


def cal_metrics(file):
    # file: {model}_result
    merged_results = list()
    f = open(f"{DATA_PATH}/test_results/{file}.json", 'r')
    for line in f.readlines():
        merged_results.extend(json.loads(line))
    
    label_convert = {"pos": 1, "neg": 0}
    pred = [label_convert[_["predict"]] for _ in merged_results]
    label = [label_convert[_["label"]] for _ in merged_results]
    pred_score = [_["prob"] for _ in merged_results]

    id_ = [_["Issue_Url"] for _ in merged_results]

    metrics, fpr, tpr = model_measure(label, pred, pred_score, id_)
    print(metrics)
    fn = file.split("_")[:-1]
    fn.append("metric_all")
    fn = '_'.join(list(fn))
    
    with open(f"{DATA_PATH}/test_results/{fn}.json", 'w') as f:
        json.dump(metrics, f, indent=4)


DATA_PATH = "xxx"

if __name__ == "__main__":
    test_set = "test_project"
    model = "out_single"
    config = json.load(open("test_config_single.json", 'r'))
    package = "MemVul"
    batch_size = 512  # 64 for CNN
    cuda = 0
    config["model"]["device"] = f"cuda:{cuda}"
    seed = 2021
    weights = None  # the default is the best one
    output_metric = f"{DATA_PATH}/test_results/{model}_metric.json"
    output_results = f"{DATA_PATH}/test_results/{model}_result.json"
    
    test(archive_file=f"{DATA_PATH}/{model}/model.tar.gz", input_file=f"{DATA_PATH}/data/{test_set}.json", test_config=config, weights_file=weights, 
        output_file=output_metric, predictions_output_file=output_results, batch_size=batch_size, cuda_device=cuda, seed=seed, package=package)