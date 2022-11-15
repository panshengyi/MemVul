# MemVul

## Project Description
This replication package contains the dataset and code for our FSE 2022 paper `Automated Unearthing of Dangerous Issue Reports`.

We are the first to introduce the task of dangerous IR (i.e., IR that leaks the vulnerability information) identification. We aim to help software vendors start the vulnerability remediation process earlier, as well as helping issue tracking systems (e.g., GitHub) better manage the disclosure process of dangerout IRs.

We collect a large-scale dataset consisting of 1,221,677 IRs from 1,390 OSS in GitHub, with 3,937 CVE-referred issue reports (CIR) in total. We consider the issue reports referred by CVE records as positives, and the remaining ones as negetives.  

We conduct preliminary study to analyze the chatacteristics of CIRs.  

We propose an automated approach named MemVul for dangerous IR identification. MemVul augments networks with a memory component, which stores the external vulnerability knowledge from Common Weakness Enumeration (CWE).

## Environments

1. OS: Ubuntu

   GPU: NVIDIA GTX 3090.

2. Language: Python (v3.8)

3. CUDA: 11.2

4. Python packages:
   * [PyTorch 1.8.1+cu11](https://pytorch.org/)
   * [AllenNLP 2.4.0](https://allennlp.org/)
   * [Transformers 4.1.0](https://huggingface.co/)
   
   Please refer the official docs for the use of these packages (especially **AllenNLP**).

5. Setup:

   We use the approach proposed by Wu *et al.* ([Data Quality Matters: A Case Study on Data Label Correctness for Security Bug Report Prediction, TSE 2021](https://ieeexplore.ieee.org/document/9371393)) in security bug prediction as our baselines (including five simple text classification approaches, RF, NB, MLP, LR and KNN). The replication package of their work is archived at [link](https://github.com/NWPU-IST/sbrbench). We directly use their implementation since they perform a customized preprocessing for text tokenization and dimension reduction. You can find the code [here](./Baseline/).
   
   We also adopt TextCNN as a neural baseline. You can find the code [here](./TextCNN/). For the training of TextCNN, we use glove embedding. Please download [Glove](http://nlp.stanford.edu/data/glove.6B.zip) first, then unzip this file and put `glove.6B.300d.txt` into the folder.

   We use [bert-base-uncased](https://huggingface.co/bert-base-uncased) from HuggingFaces Transformer Libarary. You don't need to download the pretrained model by yourself as it will be downloaded the first time you run the code. Note that we perform Masked language Modeling (MLM) to further pretrain the BERT using our collected issue report data. You can find the code for further pretraining [here](run_mlm_wwm.py). The code is modified from [the one provided by the HuggingFaces](https://github.com/huggingface/transformers/tree/master/examples/research_projects/mlm_wwm). You can find more details there.

## Dataset

**Note that all datasets are in [Google Drive](https://drive.google.com/drive/folders/1K_x6zu80aVwr4CF0Jx8_bBx53FgX7Rs2?usp=sharing)**

* `all_samples.csv`: all the **original** issue report (IR) data collected from GHArchive, consisting of 1,221,677 IRs from 1,390 OSS in GitHub, with 3,937 CVE-referred issue reports (CIR) in total. We already combine the information from the correspoding CVE records with IRs. We label IRs referred by CVE records as positives (CIR), and the remaining ones as negatives (NCIR). Below, shows an example: 

   ```
   {
    "Issue_Url": "https://github.com/Mailu/Mailu/issues/1354",
    "Issue_Created_At": "2020-02-08T09:30:20Z",
    "Issue_Title": "WARNING Fetchmail security update to all branches, update ASAP",
    "Issue_Body": "On NUMBERTAG we discovered a bug in the Fetchmail script for Mailu that has serious security consequences. ...",
    "CVE_ID": "CVE-2020-5239",
    "Published_Date": "2020-02-13T01:15Z",
    "Security_Issue_Full": "1",
   }
   ```

   *Issue_Url* is the url of the issue report, you can use it as the unique id of the issue report.

   *Issue_Created_At* is the issue creation time.

   *Issue_Title* is the **original** title of the issue.

   *Issue_Body* is the **original** body of the issue (the very first comment).

   *CVE_ID* is the id of the corresponding CVE record (if it is a CIR).

   *Published_Date* is the disclosure date of the corresponding CVE record.

   *Security_Issue_Full* is the label indicating whether it is a CIR(1) or a NCIR(0).

* `test_project.json`: test set used in the experiments. We randomly sample 10% projects and use IRs from these projects as the test set. *Note that we already replace the special tokens in issue title and body, and convert the csv file to json file*. You can find the code used to divide the dataset in [utils.py](utils.py).
   
* `train_project.json`: train set used in the experiments.

* `validation_project.json`: validation set used in the experiments.

## File Organization
There are several files and three directoris (`Baseline` - baselines from Wu *et al.*, `TextCNN` - neural baseline, `MemVul` - our proposed memory network).

### Files

* `run_mlm_wwm.py`: script for further pretraining of BERT.
* `further_pretrain.json`: config (settings) for further pretraining of BERT.

* `predict_memory.py`: You should use it for evaluations of MemVul, MemVul-o (without online negative sampling) and MemVul-p (without further pretraining). These models have an external memory, and predict an issue report by matching it with each anchor stored in the external memory.
* `test_config_memory.json`: config for test of MemVul, MemVul-o and MemVul-p.
* `predict_single.py`: You should use it for evaluations of MemVul-m (without the external memory) and TextCNN. These models directly map an input issue report to its class.
* `test_config_single`: config for test of MemVul-m
* `test_config_cnn`: config for test of TextCNN

* `utils.py`: util fuctions: e.g., divide the dataset into training set and testing set, generate the golden anchors, build CWE tree


### Directories

* `Baseline/`: code for five simple text classification approaches from Wu *et al.*'s study.

   * `baseline.py`: implementation of five simple text classification approaches.
   * `dimention_reduce.py`: custom implementation for dimention reduce.
   * `model_measure_new.py`: metrics for model evaluation.

* `MemVul/`: code of MemVul, together with three variants, i.e., MemVul-m, MemVul-o, MemVul-p. *Note that MemVul-o and MemVul-p share the reader, model architecture with MemVul. You only need to change the config file.*

   * `reader_memory.py`: dataset reader for MemVul
   * `model_memory.py`: model architecture for MemVul
   * `config_memory.json`: config for the training of MemVul
   * `config_no_online.json`: config for the training of MemVul-o
   * `config_no_pretrain.json`: config for the training of MemVul-p
   * `reader_single.py`: dataset reader for MemVul-m
   * `model_single.py`: model architecture for MemVul-m
   * `config_single.json`: config for the training of MemVul-m
   * `util.py`: util fuctions, e.g., replacement of special tokens
   * `custom_PTM_embedder.py`: custom embedder for loading our further pretrained BERT encoder
   * `custom_trainer.py`: custom trainer for supporting custom callbacks
   * `custom_metric.py`: custom validation
   * `callbacks.py`: callbacks used in training, implement online negative sampling and custom validation

* `TextCNN/`: code of our implementation of TextCNN
   * `reader_cnn.py`: dataset reader for TextCNN
   * `model_cnn.py`: TextCNN model
   * `config_cnn.json`: config for the training of TextCNN.

* `data/`: all the data (in json format) used in the experiments. You can build them using the correspoding fuctions in [utils.py](utils.py)

**Note that all data are in [Google Drive](https://drive.google.com/drive/folders/1K_x6zu80aVwr4CF0Jx8_bBx53FgX7Rs2?usp=sharing)**

   * `CVE_dict.json`: all the CVE records
   * `CWE_anchor_golden_project.json`: Anchors that we built for the external memory
   * `CWE_distribution.json`: number of CIRs that belong to each CWE category in our dataset
   * `1000.csv`: all the CWE entries in the Research View. You can download it from [here](https://cwe.mitre.org/data/definitions/1000.html)
   * `CWE_tree.json`: We orgnize the CWE entries in the Research View into a tree-like structure
   * `repo_info.json`: information (e.g., #Stars, #Forks) of OSS projects in our dataset

## Train & Test

For running the baselines, enter the directory `Baseline` and run `python baseline.py`

For further pretraining of BERT, you can run `python run_mlm_wwm.py further_pretrain.json`

For training of MemVul, MemVul-m, MemVul-o, MemVul-p, TextCNN (we implement these models using AllenNLP package),

Open terminal in the parent folder and run
`allennlp train <config file> -s <serialization path> --include-package <package name>`. Please refer to official docs of [AllenNLP](https://allennlp.org/) for more details.

For example, with `allennlp train MemVul/config_memory.json -s MemVul/out_memvul/ --include-package MemVul`, you can get the output folder at `MemVul/out_memvul/` and log information showed in the console.

For test, please follow the comments in [predict_memory.py](predict_memory.py) and [predict_single.py](predict_single.py). First, you run the test function to get the detailed results of each sample (saved in file `<model>_result.json`). Then, you run the cal_metrics function to get all the metrics (saved in `<model>_metric_all.json`).

## Reference
If you find this repository useful, please consider citing our paper
```
@inproceedings{pan2022automated,
   author = {Pan, Shengyi and Zhou, Jiayuan and Cogo, Filipe Roseiro and Xia, Xin and Bao, Lingfeng and Hu, Xing and Li, Shanping and Hassan, Ahmed E.},
   title = {Automated Unearthing of Dangerous Issue Reports},
   year = {2022},
   isbn = {9781450394130},
   publisher = {Association for Computing Machinery},
   address = {New York, NY, USA},
   url = {https://doi.org/10.1145/3540250.3549156},
   doi = {10.1145/3540250.3549156},
   booktitle = {Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
   pages = {834–846},
   numpages = {13},
   location = {Singapore, Singapore},
   series = {ESEC/FSE 2022}
}
```