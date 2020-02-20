'''
@Description: 
@Author: bairongz (baiyuu.cs@gmail.com)
@Date: 2020-01-03 23:05:17
@LastEditTime : 2020-01-05 14:24:40
'''

import torch
from torch import nn
from model import *
from tqdm import tqdm
import pickle

def sanity_check(args):
    pass


def show_training_profile(args, model):
    print(args)
    print(model)


def save_ckpt(path, args, model, optim, lang):
    """save checkpoint for warmstart, evaluation.
    """
    print("saving model to %s" % path)
    torch.save({
        "args": args,
        'model_state_dict': model.state_dict(),
        # 'optim_state_dict': optim.state_dict(),
        "lang": lang
    }, path)


def load_ckpt(path, with_optim):
    """load checkpoint for warm start, evaluation.
    """
    print("loading model from %s" % path)
    ckpt = torch.load(path)
    args = ckpt["args"]
    print(args)

    model = load_model(args)
    model.load_state_dict(ckpt["model_state_dict"])
    optim=None
    if with_optim:
        optim.load_state_dict(ckpt["optim_state_dict"])

    lang = ckpt["lang"]
    return args, model, optim, lang


def load_model(args):
    if args.model_type == "trs":
        print("creating vanilla Transformer model...")
        model = Transformer(args.n_src_vocab,
                            args.d_model,
                            args.n_head,
                            args.n_enc_layers,
                            args.n_dec_layers,
                            args.d_ff,
                            share_word_embedding=args.share_embed,
                            n_dec_vocab=args.n_trg_vocab,
                            dropout=args.dropout)


    elif args.model_type == "ut":
        print("creating Universal Transformer model...")
        if args.act:
            print("Use ACT type: %s, epslion: %f" % (args.act, args.act_eps))
        model = UniversalTransformer(args.n_src_vocab,
                                     args.d_model,
                                     args.n_head,
                                     args.n_enc_layers,
                                     args.n_dec_layers,
                                     args.d_ff,
                                     share_word_embedding=args.share_embed,
                                     n_dec_vocab=args.n_trg_vocab,
                                     act=args.act,
                                     epslion=args.act_eps,
                                     )
    else:
        raise ValueError("unknown model_type: %s" % args.model_type)

    return model


def init_weight(args, model):
    def weight_init(m):
        init_method = None
        if args.init == "normal":
            init_method = torch.nn.init.normal_
        elif args.init=="xavier_uniform":
            init_method = torch.nn.init.xavier_uniform_
        elif args.init == "xavier_normal":
            init_method = torch.nn.init.xavier_normal_
        elif args.init == "kaiming_uniform":
            init_method = torch.nn.init.kaiming_uniform_
        elif args.init == "kaiming_normal":
            init_method = torch.nn.init.kaiming_normal_
        elif args.init == "orthogonal":
            init_method = torch.nn.init.orthogonal_
        else:
            raise ValueError("Unknown weight initialization method: %s"%args.init)

        if isinstance(m, (nn.Linear)):
            init_method(m.weight.data)
            if m.bias is not None and args.init not in ["xavier_uniform", "xavier_normal", "kaiming_normal", "kaiming_uniform", "orthogonal"]:
                init_method(m.bias.data)
            
            
    model.apply(weight_init)


def train_step(model, batch, loss_fn, optim, norm=0.5, act_loss_weight=0, device="cpu"):
    model.train()
    src, trg, src_pos, trg_pos, trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask \
        = list(map(lambda x: x.to(device), batch))

    dec_src = trg[:-1].detach().contiguous()
    trg = trg[1:].detach().contiguous()
    logits = model(src, dec_src, src_pos, trg_pos, trg_mask=trg_mask,
                   src_key_padding_mask=src_key_padding_mask,
                   trg_key_padding_mask=trg_key_padding_mask,
                   memory_key_padding_mask=memory_key_padding_mask
                   )

    loss = loss_fn(logits, trg.view(-1))
    if isinstance(model, UniversalTransformer) and act_loss_weight:
        loss += act_loss_weight * model.act_loss()

    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), norm)
    optim.step()

    return loss.item()


def valid_step(model, valid_iter, loss_fn, device="cpu"):
    model.eval()

    loss_record = []
    with torch.no_grad():
        for batch in valid_iter:
            src, trg, src_pos, trg_pos, trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask \
                = list(map(lambda x: x.to(device), batch))

            dec_src = trg[:-1].detach().contiguous()
            trg = trg[1:].detach().contiguous()
            logits = model(src, dec_src, src_pos, trg_pos, trg_mask=trg_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           trg_key_padding_mask=trg_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask
                           )

            loss = loss_fn(logits, trg.view(-1))
            loss_record.append(loss.item())

    return np.mean(loss_record)

