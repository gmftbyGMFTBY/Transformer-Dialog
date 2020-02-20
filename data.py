'''
@Description: data i/o related
@Author: bairongz (baiyuu.cs@gmail.com)
@Date: 2019-10-07 15:53:21
@LastEditTime : 2020-01-03 23:16:55
'''
import torch
from torch.utils.data import Dataset, DataLoader

import pickle

import constants as C
import os
import random
from random import shuffle

from collections import Counter
from torchtext import vocab
from data_utils import *

class Lang:
    """build vocabulary, 
    update@20191203: act as a wrapper for torchtext.vocab.Vocab"""

    def __init__(self, corpus, reserved_tokens=[], n_vocab=None, min_freq=1, type="paired"):
        """extract vocabulary form given corpus


        Keyword Arguments:
            reserved_tokens {list} -- reserved tokens list, will be added after the built-in tokens (default: {[]})
            n_vocab {int} -- # of max vocabulary based on the frequency, -1 means infinity.  (default: {-1})
            type {str} -- value in {"dialog", "translation"}. (default: {"dialog"})
        """
        if n_vocab == -1:
            n_vocab = None

        reserved_tokens = ["<pad>", "<unk>",
                           "<bos>", "<eos>", ] + reserved_tokens
        self.vocab = vocab.Vocab(self._build_vocab(
            corpus, type=type), specials=reserved_tokens, max_size=n_vocab, min_freq=min_freq)


    def __len__(self):
        return len(self.vocab)

    @property
    def size(self):
        return len(self.vocab)

    def idx2toks(self, idx_seq):
        return list(map(
            lambda i: self.vocab.itos[i], idx_seq
        ))

    def idx2str(self, idx_seq, spliter=" "):
        words = self.idx2toks(idx_seq)
        return spliter.join(words)

    def toks2idx(self, tok_seq):
        return list(map(
            lambda k: self.vocab.stoi[k] if k in self.vocab.stoi else self.vocab.stoi["<unk>"], tok_seq
        ))

    def add(self, token):
        raise NotImplementedError

    def _build_vocab(self, corpus, type="paired"):
        """count word occurance in the given corpus

        Arguments:
            corpus {list} -- corpus with elements in word-level
            type {str} - value in {"dialog","translation"}

        Returns:
            collection.Counter object
        """
        vocab_counter = Counter()
        if type == "paired":
            for src, trg in corpus:
                vocab_counter.update(src)
                vocab_counter.update(trg)

        elif type in ["translation", "single"]:
            for line in corpus:
                vocab_counter.update(line)
        else:
            raise ValueError

        return vocab_counter


class SingleTurnDialogData(Dataset):
    def __init__(self, corpus, n_vocab=50000, cached_data=None, lang_src=None, lang_trg=None, share_embed=False):
        super().__init__()
        if share_embed:
            self.src_lang = Lang(corpus, n_vocab=n_vocab)
            self.trg_lang = self.src_lang
        else:
            if lang_src:
                self.src_lang = lang_src
            else:
                src_sentences = list(map(lambda x: x[0], corpus))
                self.src_lang = Lang(
                    src_sentences, n_vocab=n_vocab, type="translation")

            if lang_trg:
                self.trg_lang = lang_trg
            else:
                trg_sentences = list(map(lambda x: x[1], corpus))
                assert len(trg_sentences) == len(src_sentences)
                self.trg_lang = Lang(
                    trg_sentences, n_vocab=n_vocab, type="translation")

        if cached_data:
            self.data = cached_data
        else:
            data = []
            for src, trg in corpus:
                bundle = dict()
                bundle["src_txt"] = src
                bundle["src_id"] = torch.LongTensor(
                    self.src_lang.toks2idx(src))
                bundle["trg_txt"] = trg
                bundle["trg_id"] = torch.LongTensor(
                    [C.BOS]+self.trg_lang.toks2idx(trg)+[C.EOS])

                bundle["src_pos"] = torch.arange(1, 1+len(src))
                bundle["trg_pos"] = torch.arange(1, 1+len(trg)+1)

                bundle["src_txt_back"] = self.src_lang.idx2toks(
                    bundle["src_id"])
                bundle["trg_txt_back"] = self.trg_lang.idx2toks(
                    bundle["trg_id"])
                data.append(bundle)

            self.data = data

    def __getitem__(self, index):
        bundle = self.data[index]
        src_id = bundle["src_id"]
        trg_id = bundle["trg_id"]
        src_pos = bundle["src_pos"]
        trg_pos = bundle["trg_pos"]
        return src_id, trg_id, src_pos, trg_pos

    def __len__(self):
        return len(self.data)


class MachineTranslationData(Dataset):
    def __init__(self, corpus, n_vocab=50000, cached_data=None, lang_src=None, lang_trg=None, share_embed=False):
        super().__init__()
        if share_embed:
            self.src_lang = Lang(corpus, n_vocab=n_vocab)
            self.trg_lang = self.src_lang
        else:
            if lang_src:
                self.src_lang = lang_src
            else:
                src_sentences = list(map(lambda x: x[0], corpus))
                self.src_lang = Lang(
                    src_sentences, n_vocab=n_vocab, type="translation")

            if lang_trg:
                self.trg_lang = lang_trg
            else:
                trg_sentences = list(map(lambda x: x[1], corpus))
                assert len(trg_sentences) == len(src_sentences)
                self.trg_lang = Lang(
                    trg_sentences, n_vocab=n_vocab, type="translation")

        if cached_data:
            self.data = cached_data
        else:
            nmt_data = []
            for src, trg in corpus:
                bundle = dict()
                bundle["src_txt"] = src
                bundle["src_id"] = torch.LongTensor(
                    self.src_lang.toks2idx(src))
                bundle["trg_txt"] = trg
                bundle["trg_id"] = torch.LongTensor(
                    [C.BOS]+self.trg_lang.toks2idx(trg)+[C.EOS])

                bundle["src_pos"] = torch.arange(1, 1+len(src))
                bundle["trg_pos"] = torch.arange(1, 1+len(trg)+1)

                bundle["src_txt_back"] = self.src_lang.idx2toks(
                    bundle["src_id"])
                bundle["trg_txt_back"] = self.trg_lang.idx2toks(
                    bundle["trg_id"])
                nmt_data.append(bundle)

            self.data = nmt_data

    def __getitem__(self, index):
        bundle = self.data[index]
        src_id = bundle["src_id"]
        trg_id = bundle["trg_id"]
        src_pos = bundle["src_pos"]
        trg_pos = bundle["trg_pos"]
        return src_id, trg_id, src_pos, trg_pos

    def __len__(self):
        return len(self.data)


def get_dataloader(dataset, batch_size, n_vocab, cached_path=None, test_only=False, share_embed=False, is_distributed=False):
    """create train/valid/test dataloader for training/testing

    Arguments:
        dataset {str} -- dataset name currently {weibo, iwslt14}
        batch_size {int} -- batch size for train dataloader
        n_vocab {int} -- maximun # of vocabulary

    Keyword Arguments:
        cached_path {[type]} -- [description] (default: {None})
        test_only {bool} -- [description] (default: {False})

    Returns:
        dataloaders
    """
    if dataset == "weibo":
        print("loading %s data" % dataset)
        dialogs = read_raw_weibo_data(
            C.weibo_data_src_path, C.weibo_data_trg_path)
        n_valid = min(int(len(dialogs) * 0.05), 10000)
        shuffle(dialogs)

        test_dialogs = dialogs[:n_valid]
        valid_dialogs = dialogs[n_valid:n_valid*2]
        train_dialogs = dialogs[n_valid*2:]

        if cached_path:
            print("loading cached data from %s" % cached_path)
            with open(cached_path+".lang", "rb") as f:
                lang = pickle.load(f)

            if test_only:
                test_data = SingleTurnDialogData(
                    test_dialogs, cached_lang=lang)
                test_iter = DataLoader(
                    test_data, batch_size=1, shuffle=False, collate_fn=padding_for_trs)
                return test_iter

            with open(cached_path+".data", "rb") as f:
                data = pickle.load(f)

            train_data = SingleTurnDialogData(
                None, cached_dialog=data, cached_lang=lang)
            valid_data = SingleTurnDialogData(valid_dialogs, cached_lang=lang)
            test_data = SingleTurnDialogData(test_dialogs, cached_lang=lang)

        else:
            print("Making dataset...")
            train_data = SingleTurnDialogData(train_dialogs, n_vocab=n_vocab)
            valid_data = SingleTurnDialogData(
                valid_dialogs, cached_lang=train_data.lang)
            test_data = SingleTurnDialogData(
                test_dialogs, cached_lang=train_data.lang)

            os.makedirs(os.path.join(C.exp_dir, "exp_data"), exist_ok=True)
            cached_fea = "cached%d.pkl" % len(train_data)
            cached_path = os.path.join(C.exp_dir, "exp_data", cached_fea)
            print("saving data to %s" % cached_path)
            with open(cached_path+".lang", "wb") as f:
                pickle.dump(train_data.lang, f)

            with open(cached_path+".data", "wb") as f:
                pickle.dump(train_data.data, f)

        train_iter = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, collate_fn=padding_for_trs)
        valid_iter = DataLoader(valid_data, batch_size=1,
                                shuffle=False, collate_fn=padding_for_trs)
        test_iter = DataLoader(test_data, batch_size=1,
                               shuffle=False, collate_fn=padding_for_trs)

    elif dataset == "iwslt14":
        print("loading %s data from %s" % (dataset, C.iwslt14_path))
        if test_only:
            raise NotImplementedError

        train_raw, valid_raw, test_raw = read_raw_iwslt14_data(C.iwslt14_path)
        print("creating dataset...")
        train_data = MachineTranslationData(
            train_raw, n_vocab=n_vocab, share_embed=share_embed)
        valid_data = MachineTranslationData(
            valid_raw, lang_src=train_data.src_lang, lang_trg=train_data.trg_lang)
        test_data = MachineTranslationData(
            test_raw, lang_src=train_data.src_lang, lang_trg=train_data.trg_lang)

        if is_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data)
        else:
            train_sampler = None

        train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=(
            train_sampler is None), sampler=train_sampler, collate_fn=padding_for_trs)
        valid_iter = DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, collate_fn=padding_for_trs)
        test_iter = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, collate_fn=padding_for_trs)

    elif dataset == "weibo1m":
        print("loading %s data from %s" % (dataset, C.weibo1m_path))
        train_raw, valid_raw, test_raw = read_raw_weibo1m_data(C.weibo1m_path)

        print("creating dataset...")
        train_data = SingleTurnDialogData(
            train_raw, n_vocab=n_vocab, share_embed=share_embed)
        valid_data = SingleTurnDialogData(
            valid_raw, lang_src=train_data.src_lang, lang_trg=train_data.trg_lang)
        test_data = SingleTurnDialogData(
            test_raw, lang_src=train_data.src_lang, lang_trg=train_data.trg_lang)

        train_iter = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, collate_fn=padding_for_trs)
        valid_iter = DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, collate_fn=padding_for_trs)
        test_iter = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, collate_fn=padding_for_trs)

    elif dataset == "wmt14":
        print("loading %s data from %s" % (dataset, C.wmt14_path))
        if test_only:
            raise NotImplementedError

        train_raw, valid_raw, test_raw = read_raw_iwslt14_data(C.wmt14_path)
        print("creating dataset...")
        train_data = MachineTranslationData(
            train_raw, n_vocab=n_vocab, share_embed=share_embed)
        valid_data = MachineTranslationData(
            valid_raw, lang_src=train_data.src_lang, lang_trg=train_data.trg_lang)
        test_data = MachineTranslationData(
            test_raw, lang_src=train_data.src_lang, lang_trg=train_data.trg_lang)

        if is_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=(
            train_sampler is None), sampler=train_sampler, collate_fn=padding_for_trs)
        valid_iter = DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, collate_fn=padding_for_trs)
        test_iter = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, collate_fn=padding_for_trs)

    elif dataset =="random":
        print("loading data...")
        # codes for loadding train/valid/test data from raw file
        train_raw, valid_raw, test_raw = read_raw_rand_data(n_samples=100000)
        print("creating dataset...")
        train_data = MachineTranslationData(
            train_raw, n_vocab=n_vocab, share_embed=share_embed)
        valid_data = MachineTranslationData(
            valid_raw, lang_src=train_data.src_lang, lang_trg=train_data.trg_lang)
        test_data = MachineTranslationData(
            test_raw, lang_src=train_data.src_lang, lang_trg=train_data.trg_lang)


        train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=padding_for_trs)
        valid_iter = DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, collate_fn=padding_for_trs)
        test_iter = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, collate_fn=padding_for_trs)
    else:
        # print("loading data...")
        # codes for loadding train/valid/test data from raw file
        # print("creating dataset...")
        # codes for create torch.data.Dataset for train/valid/test
        # codes for create torch.data.Dataloader
        raise ValueError("Unknow dataset name: %s" % dataset)

    print("="*10 + "Sanity Check"+"="*10)
    print("# of training example: %d" % len(train_iter))
    print("# of validation example: %d" % len(valid_iter))
    print("# of test example: %d" % len(test_iter))

    print("data example:")
    print(train_data.data[4])

    return train_iter, valid_iter, test_iter
