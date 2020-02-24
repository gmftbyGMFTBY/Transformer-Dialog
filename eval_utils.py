'''
@Description: 
@Author: bairongz (baiyuu.cs@gmail.com)
@Date: 2020-01-05 15:28:52
@LastEditTime : 2020-01-05 15:29:20
'''
from nltk.translate.bleu_score import corpus_bleu


def compute_bleu(truth, hypo):
    """compute corpus bleu score

    Arguments:
        truth {list} -- list of ground truth string
        hypo {list} -- list of hypothesis string

    Returns:
        float -- bleu score
    """
    truth = [i.replace('<user0>', '').replace('<user1>', '').strip() for i in truth]
    hypo = [i.replace('<user0>', '').replace('<user1>', '').strip() for i in hypo]
    references = [[item.strip().split()]for item in truth]
    candidates = [item.strip().split() for item in hypo]
    score = corpus_bleu(references, candidates)
    return score


def metric(results, metric_type="bleu"):
    if metric_type == "bleu":
        truth = list(map(lambda x: x[1][0], results))
        hypos = list(map(lambda x: x[1][1], results))
        bleu_score = compute_bleu(truth, hypos)
        return bleu_score
    else:
        raise ValueError("Unknown metric type %s" % metric_type)
