from dataclasses import replace
from os import remove, stat
from numpy.core.fromnumeric import sort
import torch
import json
import random
import numpy as np
from math import log10, log
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, get_mask_from_sequence_lengths, sort_batch_by_length
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import WhitespaceTokenizer, SpacyTokenizer
import re
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy

from typing import List, Union
from allennlp.common import FromParams
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation

import pandas as pd


DOC_PATTERN_URL = re.compile(r"example|tutorial|doc|api|quickstart|note|guide|blog|using|reference|\.io|sample|demo|template", re.I)
DOC_PATTERN_CODE = re.compile(r"example", re.I)
ERROR_PATTERN = re.compile(r"exception|error|warning|404|can't|can\s{0,1}not|could\s{0,1}not|un[a-z]{3,}", re.I)
ISSUE_PATTERN = re.compile(r"issue|pull|#[0-9]+|[0-9]{4,}", re.I)
TAG_PATTERN = re.compile(r'CODETAG|APITAG')
CODE_PATTERN = re.compile(r'[=;%$/<>\{\}\[\]]|public\sstatic\s(void){0,1}(\smain){0,1}|String')

NUM_PATTERN = re.compile(r'^(([^a-uwyz]+?\d[^a-uwyz]*(beta[0-9]+){0,1})|(\s*beta[0-9]+))\s*$', re.I)
PATH_PATTERN = re.compile(r'^\s*([^\s\(\)]+?[/\\]){2,}?[^\s\(\)]*\s*$')

API_PATTERN = re.compile(r'^\s*\S+\s*$')
WORD_PATTERN = re.compile(r'^[a-z\s]+$', re.I)
WORD_PATTERN_1 = re.compile(r'^yaml|^\s*([a-z]+[,\.\?]?\s+)*?[a-z]+[,\.\?]?\s*$', re.I)

def replace_tokens_simple(content):
    if type(content) != str:
        print("ERROR: not str")
        content = ""
        return content

    content = re.sub(r'<!---.*?-->', ' ', content)
    MAX_API_LENGTH = 150
    for _ in re.finditer(r"```.*?```", content, flags=re.S):
        code = _.group()
        if code == "``````":
            content = content.replace(code, " ", 1)
        elif ERROR_PATTERN.search(code):
            content = content.replace(code, " ERRORTAG ", 1)
        elif WORD_PATTERN_1.search(code[3:-3]):
            content = content.replace(code, f" {code[3:-3]} ", 1)
        elif API_PATTERN.search(code[3:-3]) or len(code[3:-3]) <= MAX_API_LENGTH:
            content = content.replace(code, " APITAG ", 1)
        else:
            content = content.replace(code, " CODETAG ", 1)

    for _ in re.finditer(r'`.*?`', content, flags=re.S):
        code = _.group()
        if code == "``":
            content = content.replace(code, " ", 1)
        elif ERROR_PATTERN.search(code):
            content = content.replace(code, " ERRORTAG ", 1)
        elif WORD_PATTERN_1.search(code[1:-1]):
            content = content.replace(code, f" {code[1:-1]} ", 1)
        elif API_PATTERN.search(code[1:-1]) or len(code[1:-1]) <= MAX_API_LENGTH:
            content = content.replace(code, " APITAG ", 1)
        else:
            content = content.replace(code, " CODETAG ", 1)
    
    for _ in re.finditer(r'[!]?\[(.+?)\]\((\S+)\)', content, flags=re.S):
        hyperef = _.group()
        ref = _.group(1)
        link = _.group(2)
        if re.search(r'\.', ref[-5:-1]) or re.search(r'\.', link[-5:-1]):
            content = content.replace(hyperef, " FILETAG ", 1)
        else:
            content = content.replace(hyperef, f" {ref} {link} ", 1)

    content = re.sub(r'<[^>]*>{2,}', ' APITAG ', content)
    content = re.sub(r'<[^>]*?[!;=/$%][^>]*>', ' APITAG ', content)
    
    for _ in re.finditer(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content):
        url = _.group()
        if re.search(r'bugzilla|mitre|bugs', url, flags=re.I):
            # https://cve.mitre.org/
            # https://cwe.mitre.org/
            content = content.replace(url, " CVETAG ", 1)
        elif re.search(r'\.', url[-5:-1]):
            content = content.replace(url, " FILETAG ", 1)
        else:
            content = content.replace(url, " URLTAG ", 1)
    
    content = re.sub(r'(\\r\\n)|(\\n\\n)|(\\r\\r)|(\\t\\t)|(\\\")|(\\\')', ' ', content)
        
    content = re.sub(r'\*{1,}', ' ', content)

    content = re.sub(r'#{1,}', ' ', content)

    content = re.sub(r'CVE-[0-9]+-[0-9]+', ' CVETAG ', content)

    content = re.sub(r'CWE-[0-9]+', ' CVETAG ', content)

    # content = re.sub(r'\[.*\.[A-Za-z]{2,4}\]([^A-Za-z0-9]*IMAGETAG)*', ' IMAGETAG ', content)
    
    content = re.sub(r'[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z]{1,13}\.[com,cn,net]{1,3}', ' EMAILTAG ', content)
    
    # content = re.sub(r'^\s*@\S+|@([^A-Z\s]\S+|\S*?[0-9_\-]\S*)', ' MENTIONTAG ', content)

    content = re.sub(r'@[a-zA-Z0-9_\-]+[,\.]?\s', ' MENTIONTAG ', content)

    # content = re.sub(r'\S+?example\S*', ' DOCUMENTTAG ', content, re.I)  # iteratorExample.java dl4j-examples

    content = re.sub(r'\S+?(Error|Exception)([^A-Za-z\s]\S*|\s|$)|404', ' ERRORTAG ', content)

    # content = re.sub(r'#[0-9]+', ' ISSUETAG ', content)  # issue

    # content = re.sub(r'\s[A-Za-z]+\.(ml|xml|png|csv|jar|sh|sbt|zip|exe|md|txt|js|yml|yaml|json|sql|html|jsp|php|prod|scss|ts)[,\.]*\s+', ' FILETAG ', content, re.I)

    content = re.sub(r'([^\s\(\)]+?[/\\]){2,}[^\s\(\)]*', ' PATHTAG ', content)

    for _ in re.finditer(r'\s(\S+?\.(ml|xml|png|csv|jar|sh|sbt|zip|exe|md|txt|js|yml|yaml|json|sql|html|pdf|jsp|php|prod|scss|ts|jpg|png|bmp|gif))[?,\.]{0,1}\s', content, re.I):
        file = _.group(1)
        # if re.search(r'png|jpg|bmp', file, re.I):
        #     content = content.replace(file, " IMAGETAG ", 1)
        # else:
        content = content.replace(file, " FILETAG ", 1)
    
    content = re.sub(r'-', ' ', content)
    content = re.sub(r'\S{30,}', ' APITAG ', content)

    content = re.sub(r'\S+?((\(\))|(\[\]))\S*|[^,;\.\s]{3,}?\.\S{4,}|\S+?([a-z][A-Z]|[A-Z][a-z]{2,}?)\S*|@\S+|<\S*?>', ' APITAG ', content)

    content = re.sub(r'[^a-uwyz]+?\d[^a-uwyz]*(beta[0-9]+){0,1}|beta[0-9]+', ' NUMBERTAG ', content, flags=re.I)

    content = re.sub(r'[\r\n\t]', ' ', content)
    content = re.sub(r'(\\r)|(\\n)|(\\t)|(\\\")|(\\\')', ' ', content)

    content = ' '.join([_ for _ in content.split(' ') if _ != ''])
    return content