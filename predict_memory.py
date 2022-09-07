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
from matplotlib.colors import same_color
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
from numpy.core.defchararray import encode
from numpy.core.fromnumeric import sort
from numpy.lib.npyio import load

from sklearn import metrics
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)


def test_siamese(archive_file, input_file, input_golden_file, test_config=None, weights_file=None, output_file=None, predictions_output_file=None, batch_size=64, cuda_device=0, seed=2021, package="MemVul", batch_weight_key="", file_friendly_logging=False) -> Dict[str, Any]:
    
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    # archive_file must be model.tar.gz
    import_module_and_submodules(package)  # import modules under this package 
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
    
    dataset_reader = archive.dataset_reader  # load test samples
    dataset_reader_validation = archive.validation_dataset_reader  # load golden samples

    model.eval()

    # need to prepare the feature vectors for anchors in the external memory
    logger.info("Reading golden data from %s", input_golden_file)
    golden_samples = list(dataset_reader_validation.read(input_golden_file))
    
    model.forward_on_instances(golden_samples[:128])
    if len(golden_samples) > 128:
        model.forward_on_instances(golden_samples[128:])

    # print(model._golden_instances_embeddings.shape)
    # print(model._golden_instances_labels)

    # Load the evaluation data
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


def model_measure(test_label, pred, pred_score, sample_id):
    # test_label is the ground truth
    # pred is the predicted label
    # pred_score is the predicted score
    # label of pos is 1, label of neg is 0
    print("num of testing sample:", len(test_label))
    init_index = [0, 0, 0, 0, 0, 0, 0, 0]
    TP, FN, TN, FP, pd, pf, prec, f_measure = init_index
    for i in range(len(test_label)):
        if pred[i] == test_label[i] == 1:
            # pred=1, label=1
            TP += 1
        elif test_label[i] == 1 and pred[i] != test_label[i]:
            # pred=0, label=1
            FN += 1
        elif pred[i] == test_label[i] == 0:
            # pred=0, label=0
            TN += 1
        elif test_label[i] == 0 and pred[i] != test_label[i]:
            # pred=1, label=0
            FP += 1
    if TP + FN != 0:
        # recall
        pd = TP / (TP + FN)
    # if FP + TN != 0:
    #     pf = FP / (FP + TN)
    if TP + FP != 0:
        prec = TP / (TP + FP)
    if pd + prec != 0:
        f_measure = 2 * pd * prec / (pd + prec)

    fpr, tpr, thresholds = metrics.roc_curve(test_label, pred_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)  # x, y

    ap = metrics.average_precision_score(test_label, pred_score, pos_label=1)
    precision, recall, thresholds = metrics.precision_recall_curve(test_label, pred_score, pos_label=1)

    result = {"TP": TP, "FN": FN, "TN": TN, "FP": FP, "pd&recall": pd, "prec": prec, "f1": f_measure, "ap": ap, "auc": auc}
    
    return result, fpr, tpr


def cal_metrics(file, thres=0.5):
    # file: {model}_result
    # thres is from validation (find on the validation set)
    merged_results = list()
    f = open(f"{DATA_PATH}/test_results/{file}.json", 'r')
    for line in f.readlines():
        # result of multiple batches are segmented by \n
        merged_results.extend(json.loads(line))
    
    for sample in merged_results:
        # format of each sample: {"Issue_Url": id, "label": "neg" or CWE ID, "predict": match score with each anchor (dict)}
        probs = list(sample["predict"].values())
        vote_prob = np.max(probs)
        
        sample["prob"] = vote_prob
        if vote_prob >= thres:
            sample["predict"] = "pos"
        else:
            sample["predict"] = "neg"
    
    label_convert = {"pos": 1, "neg": 0}
    pred = [label_convert[_["predict"]] for _ in merged_results]

    label = ["neg" if _["label"] == "neg" else "pos" for _ in merged_results]
    label = [label_convert[_] for _ in label]

    pred_score = [_["prob"] for _ in merged_results]

    id_ = [_["Issue_Url"] for _ in merged_results]

    metrics, fpr, tpr = model_measure(label, pred, pred_score, id_)
    print(metrics)

    fn = file.split("_")[:-1]
    fn.append("metric_all")
    fn = '_'.join(list(fn))
    metrics["thres"] = thres
    with open(f"{DATA_PATH}/test_results/{fn}.json", 'w') as f:
        json.dump(metrics, f, indent=4)


DATA_PATH = "xxx"

if __name__ == "__main__":
    test_set = "test_project"
    golden_set = "CWE_anchor_golden_project"  # path of anchors in the external memory
    model = "out_memvul"
    config = json.load(open("test_config_memory.json", 'r'))  # test config
    weights = None  # the default is the best one
    batch_size = 512
    cuda = 0
    config["model"]["device"] = f"cuda:{cuda}"
    seed = 2021
    output_metric = f"{DATA_PATH}/test_results/{model}_metric.json"
    output_results = f"{DATA_PATH}/test_results/{model}_result.json"

    test_siamese(archive_file=f"{DATA_PATH}/{model}/model.tar.gz", input_file=f"{DATA_PATH}/data/{test_set}.json", input_golden_file=f"{DATA_PATH}/data/{golden_set}.json", 
        test_config=config, weights_file=weights, output_file=output_metric, predictions_output_file=output_results, batch_size=batch_size, cuda_device=cuda, seed=seed)