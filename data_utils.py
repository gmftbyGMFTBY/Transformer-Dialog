'''
@Description: low-level func for reading data from files
@Author: bairongz (baiyuu.cs@gmail.com)
@Date: 2020-01-03 23:05:17
@LastEditTime : 2020-01-03 23:53:07
'''

from model import Transformer
import random
import torch
import constants as C
import os

def _read_raw_separate(src_path, trg_path):
    """read data from paired file with source and targat sentence in seperated file

    Arguments:
        src_path {str} -- path for file containing source sentence
        trg_path {str} -- path for file containing target sentence

    Returns:
        list -- list where each element is a src-trg word sequence pair e.g. [(src1,trg1),[src2,trg2],...]
    """
    data = []
    with open(src_path) as src_f, open(trg_path) as trg_f:
        for src, trg in zip(src_f, trg_f):
            pair = (src.strip().split(), trg.strip().split())
            data.append(pair)

    return data


def read_raw_weibo_data(src_path, trg_path):
    dialogs = _read_raw_separate(src_path, trg_path)
    # for backward compatibale use
    dialogs = list(map(lambda x: [x], dialogs))
    return dialogs


def read_raw_iwslt14_data(path, test_only=False):
    train_path_en = os.path.join(path, "train.en")
    train_path_de = os.path.join(path, "train.de")
    valid_path_en = os.path.join(path, "valid.en")
    valid_path_de = os.path.join(path, "valid.de")
    test_path_en = os.path.join(path, "test.en")
    test_path_de = os.path.join(path, "test.de")

    if test_only:
        test_data = read_raw_weibo_data(test_path_de, test_path_en)
        return test_data

    else:
        train_data = _read_raw_separate(train_path_de, train_path_en)
        valid_data = _read_raw_separate(valid_path_de, valid_path_en)
        test_data = _read_raw_separate(test_path_de, test_path_en)
        return train_data, valid_data, test_data


def read_raw_wmt14_data(path):
    train_path_en = os.path.join(path, "train.en")
    train_path_de = os.path.join(path, "train.de")
    valid_path_en = os.path.join(path, "valid.en")
    valid_path_de = os.path.join(path, "valid.de")
    test_path_en = os.path.join(path, "test.en")
    test_path_de = os.path.join(path, "test.de")

    train_data = _read_raw_separate(train_path_en, train_path_de)
    valid_data = _read_raw_separate(valid_path_en, valid_path_de)
    test_data = _read_raw_separate(test_path_en, test_path_de)
    return train_data, valid_data, test_data


def read_raw_weibo1m_data(path, n_valid_samples=10000, n_test_samples=10000):
    src_file_path = os.path.join(path, "weibo_train_1m.src")
    trg_file_path = os.path.join(path, "weibo_train_1m.trg")
    dialogs = _read_raw_separate(src_file_path, trg_file_path)
    shuffle(dialogs)
    valid_dialogs = dialogs[:n_valid_samples]
    test_dialogs = dialogs[n_valid_samples:n_valid_samples+n_test_samples]
    train_dialogs = dialogs[n_valid_samples+n_test_samples:]
    return train_dialogs, valid_dialogs, test_dialogs


def read_raw_rand_data(n_samples,n_vocab=5000):
    train_data = [] 
    for _ in range(n_samples):
        trg = [random.randint(0,n_vocab) for _ in range(random.randint(10,30))]
        src = trg.copy()
        train_data.append((trg, src))

    valid_data = [] 
    for _ in range(min(int(0.05*n_samples), 5000)):
        trg = [random.randint(0,n_vocab) for _ in range(random.randint(10,30))]
        src = trg.copy()
        valid_data.append((trg, src))
    
    test_data = [] 
    for _ in range(min(int(0.05*n_samples), 5000)):
        trg = [random.randint(0,n_vocab) for _ in range(random.randint(10,30))]
        src = trg.copy()
        test_data.append((trg, src))
    
    return train_data, valid_data, test_data

def padding_for_trs(batch):
    items = zip(*batch)
    padded_src, padded_trg, src_pos, trg_pos = list(
        map(lambda x: torch.nn.utils.rnn.pad_sequence(x, padding_value=C.PAD), items))
    trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask = Transformer.get_masks(
        padded_src, padded_trg[:-1], PAD=C.PAD)

    return padded_src, padded_trg, src_pos, trg_pos, trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask
