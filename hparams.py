'''
@Description: define some common hparams for fast and reproducible experiments
@Author: bairongz (baiyuu.cs@gmail.com)
@Date: 2019-12-21 14:55:22
@LastEditTime : 2020-01-05 13:21:01
'''


def trs_base(args):
    """transformer base described in the paper
        28.0 wmt14, see
        https://github.com/OpenNMT/OpenNMT-py/issues/637
        https://github.com/OpenNMT/OpenNMT-tf/tree/master/scripts/wmt
    """
    # args.model_type="trs"
    args.d_embed=512
    args.d_model=512
    args.n_head=8
    args.d_ff=2048
    args.n_enc_layers=6
    args.n_dec_layers=6
    args.dropout=0.1
    args.label_smooth=0.1
    return args

def trs_small(args):
    """
    publicly available score:
    34.4, NOT excatly the same hparams setting!
    source: https://arxiv.org/pdf/1906.02762.pdf, 
   
    Result:
    ptrs@iwslt14: 
    trs@iwslt14: 33.4(beam_size=5)
    """
    args.d_embed=512
    args.d_model=512
    args.n_head=4
    args.d_ff=1024
    args.n_enc_layers=2
    args.n_dec_layers=2
    args.dropout=0.2
    args.lr=0.0005
    args.label_smooth=0.1
    args.grad_norm=0.5
    args.epoch=200
    args.init = "kaiming_normal"
    return args


def trs_iwslt(args):
    """
    public Transformer BLEU is roughly 33.3~
    SOTA BLEU is roughly 35~
    source: https://www.aaai.org/ojs/index.php/AAAI/article/view/4487

    Result:
    
    """
    args.model_type="ptrs"
    args.d_embed=512
    args.d_model=512
    args.n_head=4
    args.d_ff=1024
    args.n_enc_layers=6
    args.n_dec_layers=6
    args.dropout=0.1
    args.lr=0.0005
    args.label_smooth=0.1
    return args