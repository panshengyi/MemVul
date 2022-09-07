from json import decoder
from logging.config import valid_ident
from re import match, sub
import re
from turtle import color
from wsgiref.validate import PartialIteratorWrapper
from allennlp.data.tokenizers import tokenizer
from allennlp.data.tokenizers import pretrained_transformer_tokenizer
from pandas._config.config import describe_option
from scipy import rand
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizerFast
import pickle
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
import random
import json
import pandas as pd
from transformers.utils.dummy_pt_objects import DataCollator
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import date, datetime
from sklearn import metrics

from MemVul.util import replace_tokens_simple


def generate_dataset_mlm(file):
    # generate the trainset for run_mlm_wwm.py (further pretraining BERT)
    # each line corresponds to a issue reprot
    datasets = json.load(open(DATA_PATH + file, 'r'))
    print(len(datasets))
    descriptions = [f"{_['Issue_Title']}. {_['Issue_Body']}" for _ in datasets]
    new_file = DATA_PATH + f"train_project_mlm.txt"
    with open(new_file, 'w', encoding="utf-8") as f:
        f.write('\n'.join(descriptions))


def fix_time(t):
    t = t.strip()
    t = re.sub(r'\sUTC', 'Z', t)
    # t = t.strip()
    t = re.sub(r'\s', 'T', t)
    return t


def check_text(s):
    ss = s
    ss = re.sub(r'[^\u0000-\u007F]', ' ', ss)
    num_token = len(re.findall(r'[a-z]+', ss, re.I))
    # print(num_token)
    if num_token < 10:
        return False
    else:
        return True

def check_text_1(s):
    if re.search(r'[\u4E00–\u9FFF]', s):
        return False
    else:
        return True


def preprocess_dataset_csv(f):
    # preprocess the dataset
    df = pd.read_csv(DATA_PATH + f"{f}.csv", low_memory=False)
    # print(len(df))
    # print(len(df[(df.Issue_Title.isnull()) | (df.Issue_Body.isnull())]))
    print(len(df[(df.Issue_Title.isnull()) & (df.Issue_Body.isnull())]))  # 1

    # remove issue report with missing title and body
    df = df[~(df.Issue_Title.isnull() & df.Issue_Body.isnull())]
    # df = df[~(df.Issue_Title.isnull() | df.Issue_Body.isnull())]
    # print(len(df))
    # print(len(df[df.Security_Issue_Full == 1]))
    # print(len(df[df.Security_Issue_Full == 0]))

    # remove issues that created after the disclosure of correspoding CVE
    df.fillna("", inplace=True)
    df["project"] = df["Issue_Url"].map(extract_project)
    # print(len(set(df["project"].tolist())))
    
    # remove CIR created after the official disclosure of corresponding CVE
    df["Issue_Created_At"] = df["Issue_Created_At"].map(fix_time)
    print(len(df[(df.Security_Issue_Full == 1) & (df.Issue_Created_At > df.Published_Date)]))  # 53
    df = df[(df.Security_Issue_Full == 0) | (df.Issue_Created_At < df.Published_Date)]

    # remove projects without CIRs (since we remove some CIRs in the last step)
    df["valid"] = df.groupby('project')['Security_Issue_Full'].transform('sum')  # 0-invalid >=1-valid
    print(len(df[df.valid == 0]))  # 26421

    df = df[df.valid != 0]
    # print(len(df))
    # print(len(set(df["project"].tolist())))
    # print(len(df[df.Security_Issue_Full == 1]))
    # print(len(df[df.Security_Issue_Full == 0]))

    print("begin process...")
    df["Issue_Title"] = df["Issue_Title"].map(replace_tokens_simple)
    df["Issue_Body"] = df["Issue_Body"].map(replace_tokens_simple)
    print("finish process...")
    df.to_csv(DATA_PATH + f"{f}_processed.csv")


def extract_project(url):
    tmp = url.split('/')
    if len(tmp) != 7:
        print("ERROR" + url)
        return "ERROR"
    return f"{tmp[3]}/{tmp[4]}"


def divide_dataset_project_csv(f):
    # divide the train set and test set
    # divide the train set and validation set
    df = pd.read_csv(DATA_PATH + f"{f}.csv", low_memory=False)
    df.fillna("", inplace=True)
    df["project"] = df["Issue_Url"].map(extract_project)
    project = list(set(df["project"].tolist()))
    project.sort()
    print(len(df))
    print(len(project))

    project_selected = random.sample(project, k=int(len(project)*0.1))
    # print(project_selected)
    # print(len(project))
    df_train = df[~df.project.isin(project_selected)]
    print(len(df_train))
    print(len(set(df_train["project"].tolist())))
    print(len(df_train[df_train.Security_Issue_Full == 1]))
    print(len(df_train[df_train.Security_Issue_Full == 0]))
    del df_train["project"]

    # print("ratio:", len(df_train[df_train.Security_Issue_Full == 1]) / len(df_train))

    df_test = df[df.project.isin(project_selected)]
    print(len(df_test))
    print(len(set(df_test["project"].tolist())))
    print(len(df_test[df_test.Security_Issue_Full == 1]))
    print(len(df_test[df_test.Security_Issue_Full == 0]))
    del df_test["project"]

    # print("ratio:", len(df_test[df_test.Security_Issue_Full == 1]) / len(df_test))

    # divide into train and test
    df_train.to_csv(DATA_PATH + "train_project_all.csv", index=False)
    df_test.to_csv(DATA_PATH + "test_project.csv", index=False)
    # further divide into train and validation
    # df_train.to_csv(DATA_PATH + "train_project.csv", index=False)
    # df_test.to_csv(DATA_PATH + "validation_project.csv", index=False)


def build_CWE_tree():
    CWE = json.load(open(DATA_PATH + 'CWE_info.json', 'r'))
    CWE_tree = dict()
    for cwe in CWE:
        cwe_id = int(cwe["CWE-ID"])
        CWE_tree[cwe_id] = cwe
        CWE_tree[cwe_id]["father"] = list()
        CWE_tree[cwe_id]["children"] = list()
        CWE_tree[cwe_id]["peer"] = list()
        CWE_tree[cwe_id]["relate"] = list()
    
    for cwe_id, cwe in CWE_tree.items():
        relations = cwe['Related Weaknesses'].split("::")
        for r in relations:
            if "VIEW ID:1000" in r:
                rr = r.split(":")
                target_id = int(rr[3])
                if "ChildOf" in rr:
                    cwe["father"].append(target_id)
                    CWE_tree[target_id]["children"].append(cwe_id)
                elif "PeerOf" in rr or "CanAlsoBe" in rr:
                    cwe["peer"].append(target_id)
                    CWE_tree[target_id]["peer"].append(cwe_id)
                elif "CanPrecede" in rr or "Requires" in rr:
                    cwe["relate"].append(target_id)
                    CWE_tree[target_id]["relate"].append(cwe_id)
    
    with open(DATA_PATH + "CWE_tree.json", 'w') as f:
        json.dump(CWE_tree, f, indent=4)


def get_pos_sample():
    # extract all the positive samples and link them to the correspoding CWE
    dataset = pd.read_csv(DATA_PATH + "all_samples.csv", low_memory=False)
    dataset = dataset.fillna("")
    pos = dataset[dataset.Security_Issue_Full == 1]
    print(len(pos))
    print(len(dataset))

    pos = rm_unnamed_columns(pos)
    pos = pos.to_dict(orient="records")

    CVE = json.load(open(DATA_PATH + "CVE_dict.json", 'r'))
    for sample in pos:
        cve_id = sample["CVE_ID"]  # CVE_ID of all samples are valid
        sample['CWE_ID'] = CVE[cve_id]['CWE_ID']
        sample['CVE_Description'] = CVE[cve_id]['CVE_Description']
    
    with open(DATA_PATH + "pos_info.json", 'w') as f:
        json.dump(pos, f, indent=4)


def pos_distribution():
    count_missing_CWE = 0
    POS = json.load(open(DATA_PATH + "pos_info.json", 'r'))
    CWE = json.load(open(DATA_PATH + "CWE_tree.json", 'r'))
    CWE_distribution = dict()
    for pos in POS:
        cve_id = pos['CVE_ID']
        cwe_id = pos['CWE_ID'] or "null"  # cwe_id = None
        if not CWE_distribution.get(cwe_id):
            CWE_distribution[cwe_id] = {'abstraction': None, '#issue report': 0, '#CVE': 0, 'CVE_distribution': dict()}
            # three special CWE categories: NVD-CWE-noinfo, NVD-CWE-Other, null(CVE's CWE value is missing) 
            if cwe_id not in ["NVD-CWE-noinfo", "NVD-CWE-Other", "null"]:
                id_ = cwe_id.split('-')[1]
                if CWE.get(id_):
                    CWE_distribution[cwe_id]['abstraction'] = CWE[id_]["Weakness Abstraction"]
                else:
                    count_missing_CWE += 1
                    print(cwe_id)

        CWE_distribution[cwe_id]['#issue report'] += 1
        if not CWE_distribution[cwe_id]['CVE_distribution'].get(cve_id):
            CWE_distribution[cwe_id]['CVE_distribution'][cve_id] = 0
            CWE_distribution[cwe_id]['#CVE'] += 1
        
        CWE_distribution[cwe_id]['CVE_distribution'][cve_id] += 1
    
    print(count_missing_CWE)  # 11
    with open(DATA_PATH + "CWE_distribution.json", 'w') as f:
        json.dump(CWE_distribution, f, indent=4)


def BFS(cwe_id, CWE_tree, level=1):
    level += 1
    sub_tree = list()
    queue = [cwe_id, -1]
    while level != 0 and len(queue) != 0:
        node = str(queue.pop(0))
        if node == "-1":
            level -= 1
            if len(queue) != 0:
                queue.append(-1)
            continue
        sub_tree.append(node) 
        queue.extend(CWE_tree[node]["children"] + CWE_tree[node]["peer"] + CWE_tree[node]["relate"])
    
    return sub_tree


def remove_repeat(original_list):
    no_repeat_list = [original_list[0]]
    for sample in original_list:
        if sample not in no_repeat_list:
            no_repeat_list.append(sample)
    return no_repeat_list


def add_end_seperator(s):
    # add seperators at the end of the input sentence(s) during merge
    s = s.strip()
    if s == "":
        return s
    # if re.match(r'[a-z0-9]', s[-1], re.I) is not None:
    if re.match(r'\.', s[-1]) is None:
        # not end with dot
        s += '.'
    s += ' '
    return s


def generate_description(cwe_id, CWE_tree, CVE_dict=None, CWE_distribution=None, num_cve_per_anchor=5):
    # generate descriptions for each anchor
    description = ""
    if cwe_id not in CWE_tree:
        cwe = CWE_distribution[f"CWE-{cwe_id}"]
        cve_belong_to_cwe = list(cwe['CVE_distribution'].keys())
        # for cve_id in random.choices(cve_belong_to_cwe, k = 3*num_cve_per_anchor):
        for cve_id in random.sample(cve_belong_to_cwe, k=min(3*num_cve_per_anchor, len(cve_belong_to_cwe))):
            description += add_end_seperator(replace_tokens_simple(CVE_dict[cve_id]["CVE_Description"]))  # preprocess the CVE description
    else:
        description += add_end_seperator(CWE_tree[cwe_id]['Name'])
        description += add_end_seperator(CWE_tree[cwe_id]["Description"])
        for item in CWE_tree[cwe_id]['Common Consequences'].split("::"):
            if "SCOPE" in item:
                flag = False
                for element in item.split(':'):
                    if flag and element not in ['IMPACT', 'NOTE']:
                        description += add_end_seperator(element)
                    if element == 'IMPACT':
                        flag = True
        description += add_end_seperator(CWE_tree[cwe_id]["Extended Description"])

        # items = CWE_tree[cwe_id]["Observed Examples"].split("::")
        # if items[0] == "null":
        #     pass
        # else:
        #     examples = [_[_.find("DESCRIPTION")+12: _.find("LINK")-1] for _ in items]
        #     # for exa in random.choices(examples, k = num_cve_per_anchor):
        #     for exa in random.sample(examples, k=min(num_cve_per_anchor, len(examples))):
        #         description += add_end_seperator(exa)

    return description


def build_anchor(level=1, num_cve_per_anchor=5):
    # build the external memory
    abstr_level = {"Pillar": 1, "Class": 2, "Base": 2.5, "Variant": 3, "Compound": 3}
    
    CWE_distribution_train = json.load(open(DATA_PATH + "CWE_distribution_train.json", 'r'))  # only use the train set
    CWE_tree = json.load(open(DATA_PATH + "CWE_tree.json", 'r'))  # dict
    CVE_dict = json.load(open(DATA_PATH + "CVE_dict.json", 'r'))  # dict
    
    CWE_anchor = dict()
    for id_, cwe in CWE_distribution_train.items():
        description = ""
        if id_ == "null":
            # corresponding CVE record miss CWE value. considered as dirty data
            continue

        cwe_id = id_.split("-")[1]
        cve_belong_to_cwe = list(cwe['CVE_distribution'].keys())  # randomness
        num_cve = len(cve_belong_to_cwe)
        if cwe_id not in CWE_tree:
            # for CWE not in the Research View, only using the CVE description (11 nodes + NVD-CWE-noinfo + NVD-CWE-Other).
            # for cve_id in random.choices(cve_belong_to_cwe, k=3*num_cve_per_anchor):
            for cve_id in random.sample(cve_belong_to_cwe, k=min(3*num_cve_per_anchor, num_cve)):
                description += add_end_seperator(replace_tokens_simple(CVE_dict[cve_id]["CVE_Description"]))
        else:
            related_cwe = BFS(cwe_id, CWE_tree, level)  # level: use BFS to search how many levels of the subtree rooted at cwe_id
            related_cwe = remove_repeat(related_cwe)
            related_cwe = [(_, abstr_level[CWE_tree[_]["Weakness Abstraction"]]) for _ in related_cwe]
            related_cwe.sort(key=lambda x:x[1])  # high-level nodes first and then low-level nodes
            for _ in related_cwe:
                description += generate_description(_[0], CWE_tree, num_cve_per_anchor = num_cve_per_anchor)

            # for cve_id in random.choices(cve_belong_to_cwe, k=num_cve_per_anchor):
            for cve_id in random.sample(cve_belong_to_cwe, k=min(num_cve_per_anchor, num_cve)):
                description += add_end_seperator(replace_tokens_simple(CVE_dict[cve_id]["CVE_Description"]))
        
        CWE_anchor[id_] = description.strip()
    
    print(len(CWE_anchor))  # 129
    
    with open(DATA_PATH + "CWE_anchor_golden_project.json", 'w') as f:
        json.dump(CWE_anchor, f, indent=4)


def rm_unnamed_columns(df):
    # used before writing a csv file (make sure we do not write Unamed columns)
    print(df.columns.tolist())
    valid_columns = list()
    for col in df.columns.tolist():
        if 'Unnamed' in col:
            continue
        valid_columns.append(col)
    
    df = df[valid_columns]
    print(df.columns.tolist())
    return df


def csv_to_json(f):
    # convert csv file to json
    # df = pd.read_csv(DATA_PATH + "1000.csv", header=0, index_col=False)  # for converting CWE(Research View)
    df = pd.read_csv(DATA_PATH + f"{f}.csv", low_memory=False)
    df.fillna("", inplace=True)
    
    # exclude unamed columns(index)
    # explicitly specify colums or use rm_unnamed_colums
    # df = df[["Issue_Url", "Issue_Created_At", "Issue_Title", "Issue_Body", "CVE_ID", "Published_Date", "Security_Issue_Full"]]
    df = rm_unnamed_columns(df)

    records = df.to_dict(orient="records")

    with open(DATA_PATH + f"{f}.json", 'w') as ff:
        json.dump(records, ff, indent=4)


def rm_project_without_pos(f):
    df = pd.read_csv(DATA_PATH + f"{f}.csv", low_memory=False)
    df.fillna("", inplace=True)

    print(df["Security_Issue_Full"].value_counts())

    df["project"] = df["Issue_Url"].map(extract_project)
    project = list(set(df["project"].tolist()))
    # print(project)
    print(len(project))
    print(len(df[df.Security_Issue_Full == 1]))
    print(len(df[df.Security_Issue_Full == 0]))
    print(len(df))

    df["valid"] = df.groupby('project')['Security_Issue_Full'].transform('sum') 
    print(len(df[df.valid == 0]))
    print(len(df[df.valid == 0].drop_duplicates(subset=["project"])))

    # for p in project:
    #     if len(df[(df.project == p) & (df.Security_Issue_Full == 1)]) == 0:
    #         print("ERROR")

    df = df[df.valid != 0]
    print(len(set(df.project.to_list())))
    print(len(df[df.Security_Issue_Full == 1]))
    print(len(df[df.Security_Issue_Full == 0]))
    print(len(df))

    df[["Issue_Url", "Issue_Created_At", "Issue_Title", "Issue_Body", "CVE_ID", "Published_Date", "Security_Issue_Full"]].to_csv(DATA_PATH + f"{f}.csv")


def repo_info_stat(file):
    # stars and forks of the projects
    df = pd.read_csv(DATA_PATH + f"{file}.csv", low_memory=False)
    df.fillna("", inplace=True)
    df["project"] = df["Issue_Url"].map(extract_project)
    projects = set(df["project"].to_list())
    print(len(projects))

    repo_info = json.load(open(DATA_PATH + "repo_info.json", 'r'))  # stats of all the 1390 projects (8 are unable to retrieve)
    retrieved_projects = set(repo_info.keys())

    print("project unable to retrieve", len(projects - retrieved_projects))
    print(projects - retrieved_projects)

    projects = projects & retrieved_projects  # intersection

    star = [repo_info[p]["stargazers_count"] for p in projects]
    watch = [repo_info[p]["watchers_count"] for p in projects]
    fork = [repo_info[p]["forks_count"] for p in projects]
    subscribe = [repo_info[p]["subscribers_count"] for p in projects]

    print("star", np.median(star), np.average(star))
    print("watch", np.median(watch), np.average(watch))
    print("fork", np.median(fork), np.average(fork))
    print("subscribe", np.median(subscribe), np.average(subscribe))


def match_keyword(des):
    sec_related_word = r"(?i)(denial.of.service|\bxxe\b|remote.code.execution|\bopen.redirect|osvdb|\bvuln|\bcve\b|\bxss\b|\bredos\b|\bnvd\b|malicious|x−frame−options|attack|cross.site|exploit|directory.traversal|\brce\b|\bdos\b|\bxsrf\b|clickjack|session.fixation|hijack|advisory|insecure|security|\bcross−origin\b|unauthori[z|s]ed|infinite.loop|authenticat(e|ion)|bruteforce|bypass|constant.time|crack|credential|\bdos\b|expos(e|ing)|hack|harden|injection|lockout|overflow|password|\bpoc\b|proof.of.concept|poison|privelage|\b(in)?secur(e|ity)|(de)?serializ|spoof|timing|traversal)"
    if re.search(sec_related_word, des):
        return True
    return False


def preliminary_study_keyword_match():
    df = pd.read_csv(DATA_PATH + f"all_samples.csv", low_memory=False)
    df.fillna("", inplace=True)
    print(len(df))
    
    df["sec_keyword_title"] = df["Issue_Title"].map(match_keyword)
    df["sec_keyword_body"] = df["Issue_Body"].map(match_keyword)
    print(len(df))

    df_pos_match = df[(df.Security_Issue_Full == 1) & ((df.sec_keyword_title == True) | (df.sec_keyword_body == True))]
    df_pos_not_match = df[(df.Security_Issue_Full == 1) & (~df.sec_keyword_title) & (~df.sec_keyword_body)]
    df_neg_match = df[(df.Security_Issue_Full == 0) & ((df.sec_keyword_title == True) | (df.sec_keyword_body == True))]
    df_neg_not_match = df[(df.Security_Issue_Full == 0) & (~df.sec_keyword_title) & (~df.sec_keyword_body)]

    print("df_pos_match", len(df_pos_match))
    print("df_pos_not_match", len(df_pos_not_match), len(df[df.Security_Issue_Full == 1]) - len(df_pos_match))
    print("df_neg_match", len(df_neg_match))
    print("df_neg_not_match", len(df_neg_not_match), len(df[df.Security_Issue_Full == 0]) - len(df_neg_match))


def delta_days_IR_CVE():
    # draw the delta days between issue report creation and CVE disclosure
    df = pd.read_csv(DATA_PATH + f"all_samples.csv", low_memory=False)
    df.fillna("", inplace=True)

    df = df[df.Security_Issue_Full == 1]
    df["Issue_Created_At"] = df["Issue_Created_At"].map(fix_time)

    pos = df.to_dict(orient="records")
    num_pos = len(pos)
    cve_dict = json.load(open(DATA_PATH + "CVE_dict.json", 'r'))

    # count = 0
    distribution = [0, 0, 0, 0, 0]
    for ir in pos:
        # "2018-10-30T16:26Z"
        ir_creation = datetime.strptime(ir["Issue_Created_At"], "%Y-%m-%dT%H:%M:%SZ")
        cve_publish = datetime.strptime(ir["Published_Date"], "%Y-%m-%dT%H:%MZ")

        seconds_one_day = 3600 * 24
        time_delta = cve_publish - ir_creation
        delta_days = time_delta.days + (time_delta.seconds / seconds_one_day)  # seconds is within a day, total_seconds is the entire seconds
        if delta_days <= 0:
            distribution[0] += 1
        elif delta_days > 0 and delta_days <= 7:
            distribution[1] += 1
        elif delta_days > 7 and delta_days <= 30:
            distribution[2] += 1
        elif delta_days > 30 and delta_days <= 180:
            distribution[3] += 1
        else:
            distribution[4] += 1
    
    print(distribution)
    # print(count)
    distribution = [_ / num_pos for _ in distribution]
    print(distribution)
    x = range(len(distribution))

    plt.figure(figsize=(25, 10.5))
    plt.ylim(0, 0.45)
    plt.yticks(fontsize=35)
    plt.bar(x, distribution, width=0.7, color="gray", edgecolor='gray')
    plt.xticks(x, ["(-∞,0]", "(0,7]", "(7,30]", "(30,180]", "(180,+∞)"], fontsize=35)
    for a, b in zip(x, distribution):
        plt.text(a, b+0.01, to_percent(b), ha="center", fontsize=35)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xlabel("(NVD Disclosure Date - CVE-Referred IR Creation Date) in Days", fontsize=40)
    plt.ylabel("Percentage", rotation="vertical", fontsize=40)
    plt.savefig("delta_days_IR_CVE.pdf")


def culmulative_distribution_CWE():
    cwe_distribution = json.load(open(DATA_PATH + "CWE_distribution.json", 'r'))  # cwe distribution of the entire dataset (including 2 IRs with missing CWE)
    print("#CWE categories:", len(cwe_distribution))

    # write the data, use R to draw
    cwe_distribution = [(k, v["#issue report"]) for k, v in cwe_distribution.items()]
    cwe_distribution.sort(key=lambda x:x[1])  # minimum to maximum
    df = pd.DataFrame({'CWE': [_[0] for _ in cwe_distribution], 'IR_NUM': [_[1] for _ in cwe_distribution]})
    df.to_csv("ecdf_cwe.csv")

    ir_same_cwe = [_[1] for _ in cwe_distribution]
    culmulative_distribution = list()  # y
    num_ir_same_cwe = list()  # x
    culmulative_sum = 1  # the cwe category with minimum CIRs
    for i in range(1, len(ir_same_cwe)):
        if ir_same_cwe[i] != ir_same_cwe[i-1]:
            culmulative_distribution.append(culmulative_sum)
            num_ir_same_cwe.append(ir_same_cwe[i-1])
        culmulative_sum += 1
    culmulative_distribution.append(culmulative_sum)
    num_ir_same_cwe.append(ir_same_cwe[-1])
    
    culmulative_distribution = [_ / culmulative_sum for _ in culmulative_distribution]
    print([(x, y) for x, y in zip(num_ir_same_cwe, culmulative_distribution)])
    print(culmulative_distribution[num_ir_same_cwe.index(1)])
    print(culmulative_distribution[num_ir_same_cwe.index(5)])
    print(culmulative_distribution[num_ir_same_cwe.index(29)])
    print(culmulative_distribution[num_ir_same_cwe.index(724)])


def match_steps_to_reproduce(x, keyword):
    # keyword is a regex
    x = x or ""
    if re.search(keyword, x, flags=re.I):
        return True
    return False


def IR_with_attack_steps():
    # add in rebuttal
    dataset = pd.read_csv(DATA_PATH + "all_samples.csv", low_memory=False)
    dataset.fillna("", inplace=True)

    pos_dataset = dataset[dataset["Security_Issue_Full"] == 1]  # all the CIR data
    print(len(pos_dataset))  # 3937

    # do not add right \b because "PoCs" 
    # n? because "\\nPoC" in our dataset
    keyword = r"(?i)(\b(n)?poc|proof-of-concept|proof\sof\sconcept|steps\sto\sreproduce|steps\sto\sreplicate)"
    pos_dataset["attack_steps"] = pos_dataset["Issue_Body"].apply(match_steps_to_reproduce, keyword=keyword)

    print(len(pos_dataset[pos_dataset["attack_steps"] == True]))  # 1570

DATA_PATH = "data"

if __name__ == '__main__':
    preprocess_dataset_csv("all_samples")
    # divide_dataset_project_csv("all_samples_processed")
    # csv_to_json("train_project")
    # get_pos_sample()
    # pos_distribution()
    # build_anchor()
    # preliminary_study_keyword_match()
    # delta_days_IR_CVE()
    # culmulative_distribution_CWE()
    # IR_with_attack_steps()