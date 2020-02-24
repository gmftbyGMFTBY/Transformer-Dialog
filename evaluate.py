'''
@Description: 
@Author: bairongz (baiyuu.cs@gmail.com)
@Date: 2019-10-09 17:01:50
@LastEditTime : 2020-01-05 18:28:01
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle

from model import *
from tqdm import tqdm
import os

import argparse
import constants as C
from train import load_ckpt
from data import get_dataloader

from nltk.translate.bleu_score import corpus_bleu
from onmt.translate.beam_search import BeamSearch
from onmt.translate import GNMTGlobalScorer
# from torchnlp.metrics import get_moses_multi_bleu
from eval_utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", help="checkpoint path")
    parser.add_argument("--save", default="", help="save path")
    parser.add_argument("--cached_data", default="",
                        help="path to cached data")

    # parser.add_argument("--dataset", default="weibo", help="dataset name in {weibo, iwslt14}")
    parser.add_argument("--beam_size", "-bm", default=10,
                        type=int, help="beam size")
    parser.add_argument("--batch_size", "-bs", default=1,
                        type=int, help="batch size")
    parser.add_argument("--n_best", "-nb", default=1, type=int, help="n_best")
    parser.add_argument("--max_len", default=128,
                        type=int, help="maximum length")
    parser.add_argument("--alpha", default=1., type=float,
                        help="length penalty, larger number promote longer sentence")
    parser.add_argument("--device", default="cpu",
                        help="device to run experiments")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def evaluate(model, valid_or_test_iter, tok_enc,  device="cpu", tok_dec=None, beam_size=20, n_best=1, alpha=1., verbose=False):
    if tok_dec is None:
        tok_dec = tok_enc

    dialogs = []
    pbar = tqdm(valid_or_test_iter, total=len(valid_or_test_iter))
    for batch in pbar:
        src, trg, src_pos, trg_pos, * \
            _ = list(map(lambda x: x.to(device), batch))

        usr = src.cpu().squeeze().numpy().tolist()
        # exclude <bos> and <eos>
        truth = trg.cpu().squeeze().numpy().tolist()[1:-1]

        model.to(device)
        hypo_and_scores = beam_search(
            model, [src, src_pos], device, beam_size=beam_size, alpha=alpha, n_best=n_best)

        if verbose:
            for i, [hypo, score] in enumerate(hypo_and_scores[:n_best]):
                usr_txt = " ".join(tok_enc.idx2toks(usr))
                truth_txt = " ".join(tok_dec.idx2toks(truth))
                hypo_txt = " ".join(tok_dec.idx2toks(hypo[:-1]))
                print("[%2d/%2d]\n USR: %s\nTRG: %s\nSYS: %s\n" %
                      (i+1, n_best, usr_txt, truth_txt, hypo_txt))

        hypo, score = hypo_and_scores[0]
        usr_txt = " ".join(tok_enc.idx2toks(usr))
        truth_txt = " ".join(tok_dec.idx2toks(truth))
        hypo_txt = " ".join(tok_dec.idx2toks(hypo[:-1]))

        dialogs.append([usr_txt, [truth_txt, hypo_txt, score]])

    return dialogs


def beam_search(model, inputs, device="cpu", beam_size=5, n_best=1, max_len=128, alpha=1):
    """beam search for nn.Module object that implemented encode() and decode() method

    Arguments:
        model {nn.Module} -- torch.nn.Module object, required to implemented encode() and decode() method
        inputs {list} -- inputs for the encode() method

    """
    def length_penalty(length, alpha):
        return np.power((length + 5) / 6, alpha)

    model.eval()
    with torch.no_grad():
        src_ids, src_pos = list(map(lambda x: x.to(device), inputs))
        model.to(device)
        memory = model.encode(src_ids, src_pos)
        init_ids = torch.LongTensor([C.BOS]).repeat(1, 1).to(device)
        init_pos = torch.arange(1, 2).repeat(1, 1).to(device)
        dec_output = model.decode(init_ids, init_pos, src_ids, memory)

        # obtain output of last step
        last_step_hidden = dec_output[-1].squeeze()
        word_prob = torch.nn.functional.log_softmax(
            model.proj(last_step_hidden), dim=0)
        cand_score, cand_ids = word_prob.topk(beam_size)

        hypo_list = []
        for c_scr, c_ids in zip(cand_score, cand_ids):
            hypo_list.append(
                ([c_ids.item()], c_scr.item())
            )

        rst = []
        hypo_list = sorted(hypo_list, key=lambda x: x[1], reverse=True)

        while hypo_list and len(hypo_list[0][0]) < max_len and len(hypo_list) > 0:
            hypo_temp = []
            for hypo, score in hypo_list:
                # prepare decoder input
                trg_ids = torch.LongTensor(hypo).repeat(
                    1, 1).permute(1, 0).to(device)
                trg_pos = torch.arange(
                    1, 1+len(hypo)).repeat(1, 1).permute(1, 0).to(device)
                dec_output = model.decode(trg_ids, trg_pos, src_ids, memory)
                # get last step hidden state then predict word distribution
                last_step_hidden = dec_output[-1].squeeze()
                assert last_step_hidden.dim() == 1
                word_prob = torch.nn.functional.log_softmax(
                    model.proj(last_step_hidden), dim=0)
                # get topk words and its log score
                cand_score, cand_ids = word_prob.topk(beam_size)
                for c_scr, c_ids in zip(cand_score, cand_ids):
                    # new_score = score + c_scr.item()
                    # score hypo sentence with length penalty
                    new_score = (score * length_penalty(len(hypo), alpha) +
                                 c_scr.item()) / length_penalty(len(hypo)+1, alpha)
                    new_hypo = hypo + [c_ids.item()]
                    hypo_temp.append(
                        (new_hypo, new_score)
                    )

            hypo_temp = sorted(hypo_temp, key=lambda x: x[1], reverse=True)
            hypo_temp = hypo_temp[:beam_size]
            next_hypo_list = []
            for hypo, score in hypo_temp:
                if hypo[-1] == C.EOS:
                    rst.append(
                        (hypo, score)
                    )
                else:
                    next_hypo_list.append(
                        (hypo, score)
                    )

            hypo_list = next_hypo_list

        rst = sorted(rst, key=lambda x: x[1], reverse=True)
        rst.extend(hypo_list)
        return rst[:n_best]


def batch_evaluate(model, test_iter, lang_src, lang_trg=None, device="cpu", max_len=128, beam_size=20, n_best=1, alpha=1., verbose=False, sample_file='samples/sample.txt'):
    if not lang_trg:
        lang_trg = lang_src

    dialogs = []
    pbar = tqdm(test_iter, total=len(test_iter))
    if verbose:
        print(f'write samples into file {sample_file}')
        sample_file = open(sample_file, 'w')
    for inputs in pbar:
        hypos = batch_beam_search_trs(model, inputs, test_iter.batch_size, device=device,max_len=max_len, beam_size=beam_size, n_best=n_best, alpha=alpha)
        srcs = inputs[0].permute(1, 0).cpu().tolist()
        trgs = inputs[1].permute(1, 0).cpu().tolist()
        # note that the result of last batch automatically padded with meaningless data for full-shape batch
        for src, trg, n_best_hypo in zip(srcs, trgs, hypos):
            if C.PAD in src:
                src = src[:src.index(C.PAD)]
            src_txt = lang_src.idx2str(src)

            if C.PAD in trg:
                trg = trg[:trg.index(C.PAD)]
            trg_txt = lang_trg.idx2str(trg[1:-1])

            hypo_txt, score = lang_trg.idx2str(
                n_best_hypo[0][0]), n_best_hypo[0][1]
            hypo_txt = hypo_txt.replace("<eos>", "").strip()

            dialogs.append(
                [src_txt, [trg_txt, hypo_txt, score]]
            )

            if verbose:
                # write into the sample
                sample_file.write(f'-src: {src_txt}\n')
                sample_file.write(f'-trg: {trg_txt}\n')
                sample_file.write(f'-hyp: {hypo_txt}\n\n')
                
                print("User: %s" % src_txt)
                print("Target: %s" % trg_txt)
                for i, [hypo, score] in enumerate(n_best_hypo):
                    hypo_txt = lang_trg.idx2str(hypo)
                    hypo_txt = hypo_txt.replace("<eos>", "").strip()
                    print("[%d/%d] Sys: %s, Score: %f\n" %
                          (i+1, len(n_best_hypo), hypo_txt, score))
    if verbose:
        sample_file.close()

    return dialogs


def batch_beam_search_trs(model, inputs, batch_size, device="cpu", max_len=128, beam_size=20, n_best=1, alpha=1.):
    """ beam search with batch input for Transformer model

    Arguments:
        beam {onmt.BeamSearch} -- opennmt BeamSearch class
        model {torch.nn.Module} -- subclass of torch.nn.Module, required to implement .encode() and .decode() method
        inputs {list} -- list of torch.Tensor for input of encode()

    Keyword Arguments:
        device {str} -- device to eval model (default: {"cpu"})

    Returns:
        result -- 2D list (B, N-best), each element is an (seq, score) pair
    """
    beam = BeamSearch(beam_size, batch_size,
                    pad=C.PAD, bos=C.BOS, eos=C.EOS,
                    n_best=n_best,
                    mb_device=device,
                    global_scorer=GNMTGlobalScorer(alpha, 0.1, "avg", "none"),
                    min_length=0,
                    max_length=max_len,
                    ratio=0.0,
                    memory_lengths=None,
                    block_ngram_repeat=False,
                    exclusion_tokens=None,
                    stepwise_penalty=True,
                    return_attention=False,
                    )
    model.eval()
    is_finished = [False] * beam.batch_size
    with torch.no_grad():
        src_ids, _, src_pos, _, _, src_key_padding_mask, _, original_memory_key_padding_mask = list(
            map(lambda x: x.to(device), inputs))
        
        if src_ids.shape[1]!=batch_size:
            diff = batch_size - src_ids.shape[1]
            src_ids = torch.cat([src_ids] + [src_ids[:,:1]] * diff, dim=1)
            src_pos = torch.cat([src_pos] + [src_pos[:,:1]] * diff, dim=1)
            src_key_padding_mask = torch.cat([src_key_padding_mask]+[src_key_padding_mask[:1]]* diff, dim=0)
            original_memory_key_padding_mask = torch.cat([original_memory_key_padding_mask] +[original_memory_key_padding_mask[:1]]*diff, dim=0)


        model.to(device)
        original_memory = model.encode(src_ids, src_pos, src_key_padding_mask=src_key_padding_mask)

        memory = original_memory
        memory_key_padding_mask = original_memory_key_padding_mask
        while not beam.done:
            len_decoder_inputs = beam.alive_seq.shape[1]
            dec_pos = torch.arange(1, len_decoder_inputs+1).repeat(beam.alive_seq.shape[0], 1).permute(1, 0).to(device)

            # unsqueeze the memory and memory_key_padding_mask in B dim to match the size (BM*BS)
            repeated_memory = memory.repeat(1, 1, beam.beam_size).reshape(
                memory.shape[0], -1, memory.shape[-1])
            repeated_memory_key_padding_mask = memory_key_padding_mask.repeat(
                1, beam.beam_size).reshape(-1, memory_key_padding_mask.shape[1])

            decoder_outputs = model.decode(beam.alive_seq.permute(1, 0), dec_pos, _, repeated_memory, memory_key_padding_mask=repeated_memory_key_padding_mask)[-1]
            if hasattr(model, "proj"):
                logits = model.proj(decoder_outputs)
            elif hasattr(model, "gen"):
                logits = model.gen(decoder_outputs)
            else:
                raise ValueError("Unknown generator!")

            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            beam.advance(log_probs, None)
            if beam.is_finished.any():
                beam.update_finished()

                # select data for the still-alive index
                for i, n_best in enumerate(beam.predictions):
                    if is_finished[i] == False and len(n_best) == beam.n_best:
                        is_finished[i] = True

                alive_example_idx = [i for i in range(
                    len(is_finished)) if not is_finished[i]]
                if alive_example_idx:
                    memory = original_memory[:, alive_example_idx, :]
                    memory_key_padding_mask = original_memory_key_padding_mask[alive_example_idx]

    # packing data for easy accessing
    results = []
    for batch_preds, batch_scores in zip(beam.predictions, beam.scores):
        n_best_result = []
        for n_best_pred, n_best_score in zip(batch_preds, batch_scores):
            assert isinstance(n_best_pred, torch.Tensor)
            assert isinstance(n_best_score, torch.Tensor)
            n_best_result.append(
                (n_best_pred.tolist(), n_best_score.item())
            )
        results.append(n_best_result)

    return results


def write_to_file(path, dialogs):
    hypos = list(map(lambda x: x[1][1], dialogs))
    with open(path, "w+") as f:
        for hypo in hypos:
            f.write(hypo+"\n")


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    train_args, model, _, lang = load_ckpt(args.ckpt, None)
    _, _, test_iter = get_dataloader(train_args.dataset, args.batch_size, -1,
                                     cached_path=args.cached_data, share_embed=train_args.share_embed)

    if args.batch_size >= 1:
        if isinstance(lang, list):
            src_lang, dec_lang = lang
            print("# of source vocabulary: %d" % len(src_lang))
            print("# of target vocabulary: %d" % len(dec_lang))
            results = batch_evaluate(model, test_iter, src_lang, lang_trg=dec_lang, device=args.device, max_len=args.max_len,
                                     beam_size=args.beam_size, n_best=args.n_best, alpha=args.alpha, verbose=args.verbose)
        else:
            print("# of vocabulary: %d" % len(lang))
            results = batch_evaluate(model, test_iter, lang, lang_trg=None, device=args.device, max_len=args.max_len,
                                     beam_size=args.beam_size, n_best=args.n_best, alpha=args.alpha, verbose=args.verbose)

    else:
        if isinstance(lang, list):
            src_lang, dec_lang = lang
            print("# of source vocabulary: %d" % len(src_lang))
            print("# of target vocabulary: %d" % len(dec_lang))
            results = evaluate(model, test_iter, src_lang, args.device, tok_dec=dec_lang,
                               beam_size=args.beam_size, n_best=args.n_best, alpha=args.alpha, verbose=args.verbose)
        else:
            print("# of vocabulary: %d" % len(lang))
            results = evaluate(model, test_iter, lang, args.device, beam_size=args.beam_size,
                               n_best=args.n_best, alpha=args.alpha, verbose=args.verbose)

    if args.save:
        write_to_file(args.save, results)

    truth = list(map(lambda x: x[1][0], results))
    hypos = list(map(lambda x: x[1][1], results))
    print("computing BLEU score...")
    bleu_score = compute_bleu(truth, hypos)
    print("BLEU: %f" % bleu_score)
