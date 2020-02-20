'''
@Description: 
@Author: bairongz (baiyuu.cs@gmail.com)
@Date: 2019-10-07 16:58:50
@LastEditTime : 2020-01-05 16:07:18
'''
import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
# import data
from data import get_dataloader, Lang


import numpy as np
from model import *
import hparams
import constants as C
import os
from tqdm import tqdm
import pickle
import argparse
import time

from common_layers import CEWithLabelSmoothing, ScheduledOptim, Noam
from train_utils import *
from evaluate import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="tag to identify experiments")
    parser.add_argument("--device", default="cpu", help="device to run the experiments, default:cpu")
    parser.add_argument("--dataset","-d", default="iwslt14", choices=["weibo","iwslt14","weibo1m","random"], help="dataset to train the model")
    parser.add_argument("--model_type", default="trs", choices=["ut","trs","ptrs"], help="model type used in the experiments")

    parser.add_argument("--init", default="", choices=["","normal","orthogonal","kaiming_normal","kaiming_uniform","xavier_normal","xavier_uniform"], help="weight initialization method")
    parser.add_argument("--batch_size", "-bsz", default=32, type=int, help="batch size for training")
    parser.add_argument("--epoch", default=200, type=int, help="# of training epochs")
    parser.add_argument("--lr", default=1., type=float, help="learning rate ")

    parser.add_argument("--warm_start", default="", help="checkpoint path for warm start")

    parser.add_argument("--n_vocab", default=10000, type=int, help=" # of vocabulary")
    parser.add_argument("--d_embed", default=256, type=int, help="hidden size of embedding layer")
    parser.add_argument("--d_model", default=256, type=int, help="hidden size of the model")
    parser.add_argument("--n_head", default=4, type=int, help="# of multi-head attention heads")
    parser.add_argument("--n_enc_layers", default=2, type=int, help="# of encoder blocks")
    parser.add_argument("--n_dec_layers", default=2, type=int, help="# of decoder blocks")
    parser.add_argument("--d_ff", default=1024, type=int, help="dim of position-wise feedforward networks")
    parser.add_argument("--dropout", default=0.2, type=float, help="dropout rate")
    parser.add_argument("--act", default="", type=str,choices=["t2t"], help="ACT type")
    parser.add_argument("--act_eps",default=0.1, type=float, help="ACT epslion")
    parser.add_argument("--act_loss_weight", type=float, default=0,help="")

    parser.add_argument("--label_smooth", default=0, type=float, help="label smoothing factor")
    parser.add_argument("--share_embed",action="store_true", help="share word embedding between encoder and decoder")
    
    parser.add_argument("--optim", default="adam",choices=["adam","adamw"], help="optimizer")
    parser.add_argument("--lr_decay", default="noam", choices=["noam"], help="learning rate decay method")
    parser.add_argument("--grad_norm", default=0.5, type=float, help="")
    parser.add_argument("--warmup", default=4000, type=int, help="warmup step")
    parser.add_argument("--data_cache", default="", help="path for cached data")
  
    parser.add_argument("--patience", default=-1, type=int, help="max patience for early stopping. -1 to disable")
    # MISC
    parser.add_argument("--setting","-s", default="", choices=["","trs_base","trs_small", "ut"], help="predifined hyperparameter setting")
    parser.add_argument("--debug", action="store_true",help="debug mode, do not save any checkpoints")

    return parser.parse_args()

def main(args):
    if args.tag:
        args.tag="exp"+time.strftime("%m%d%H%M%S",time.localtime())+"-"+args.tag
    else:
        args.tag = "exp"+time.strftime("%m%d%H%M%S",time.localtime())

    if args.setting:
        funcs = dir(hparams)
        if args.setting not in funcs:
            raise ValueError("Unknown setting %s"%args.setting)
        
        print("Loading predefined hyperparameter setting %s"%args.setting)
        args = getattr(hparams, args.setting)(args)

    print(args)
    train_iter, valid_iter, test_iter = get_dataloader(args.dataset, args.batch_size, n_vocab=args.n_vocab, cached_path=args.data_cache,share_embed=args.share_embed)
    n_vocab=None
    if args.share_embed:
        n_src_vocab = train_iter.dataset.src_lang.size
        n_trg_vocab = n_src_vocab
        print("# of vocabulary: %d"%n_src_vocab)
    else:
        n_src_vocab = train_iter.dataset.src_lang.size
        n_trg_vocab = train_iter.dataset.trg_lang.size
        print("# of source vocabulary: %d"%n_src_vocab)
        print("# of target vocabulary: %d"%n_trg_vocab)
    
    args.n_src_vocab = n_src_vocab
    args.n_trg_vocab = n_trg_vocab

    model = load_model(args)
    if args.init:
        print("apply weight initialization method: %s"%args.init)
        init_weight(args, model)
        
    if args.label_smooth>0:
        print("using Cross Entropy Loss with Label Smoothing factor %f"%args.label_smooth)
        loss_fn = CEWithLabelSmoothing(n_trg_vocab,label_smoothing=args.label_smooth, ignore_index=C.PAD)
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=C.PAD)
    else:
        print("using Cross Entropy Loss")
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=C.PAD)
        ce_loss_fn = loss_fn
    
    # optimizer related stuff
    if args.optim=="adam":
        print("Optimizer: ", args.optim)
        optim = torch.optim.Adam(model.parameters(),lr=args.lr)
    elif args.optim =="adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown optimizer type: %s"%args.optim)

    # learning rate scheduler realated stuff
    if args.lr_decay == "noam":
        print("Learning rate scheduler: ",args.lr_decay)
        scheduler = Noam(optim, args.warmup, args.d_model)
    elif args.lr_decay =="none":
        scheduler = torch.optim.lr_scheduler.StepLR(optim, float("inf"))
    else:
        raise ValueError("Unknown lr dacay type: %s"%args.lr_decay)
    
    if args.warm_start:
        args, model, optim = load_ckpt(args.warm_start, optim)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True)
    
    # if args.warm_start:
    #     args, model, optim = load_ckpt(args.warm_start, optim)

    model.to(args.device)

    # show_training_profile(args, model)
    
    best_valid_loss = float("inf")
    best_ckpt=-1
    best_ckpt_path = os.path.join(C.model_save_path, args.tag, "ckpt.ep.best")
    patience_cnt=0
    print("="*10+"start_training"+"="*10)
    for i_epoch in range(args.epoch):
        if args.patience!=-1 and patience_cnt >= args.patience:
            print("MAX PATIENCE REACHED!")
            break

        # loss_record=[]
        pbar = tqdm(train_iter, total=len(train_iter))
        cnt=0
        for i, batch in enumerate(pbar):
            loss = train_step(model, batch, loss_fn, optim, device=args.device, act_loss_weight=args.act_loss_weight, norm=args.grad_norm)
            lr = optim.param_groups[0]["lr"]

            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                ppl = np.exp(loss)
                pbar.set_description("[%3d/%3d] PPL: %f, lr: %.6f"%(i_epoch,args.epoch, ppl, lr))
            elif isinstance(loss_fn, CEWithLabelSmoothing):
                pbar.set_description("[%3d/%3d] KL: %f, lr: %.6f"%(i_epoch,args.epoch,loss, lr))
            
            scheduler.step()
            # tensorboard related stuff
            # writer.add_scalar('training loss', loss, i_epoch * len(train_iter) + i)
            # writer.add_scalar('training ppl', ppl, i_epoch * len(train_iter) + i)

        # ppl = np.exp(np.mean(loss_record))
        # print("Epoch: [%2d/%2d], Perplexity: %f, loss: %f"%(i_epoch, args.epoch, ppl, np.mean(loss_record)))

        if not args.debug:
            os.makedirs(os.path.join(C.model_save_path, args.tag), exist_ok=True)
            model_save_path = os.path.join(C.model_save_path, args.tag, "ckpt.ep.%d"% i_epoch)
            if args.share_embed:
                save_ckpt(model_save_path, args, model, optim, train_iter.dataset.src_lang)
            else:
                save_ckpt(model_save_path,args,model,optim,[train_iter.dataset.src_lang, train_iter.dataset.trg_lang])
            
            print("model saved at %s"%model_save_path)

        valid_loss = valid_step(model, valid_iter, ce_loss_fn, device=args.device)
        if not args.debug and valid_loss < best_valid_loss:
            print("new best checkpoint!")
            if os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            os.symlink("ckpt.ep.%d"%i_epoch, best_ckpt_path)
            best_valid_loss = valid_loss
            best_ckpt = i_epoch
            patience_cnt=0
        else:
            patience_cnt+=1

        if isinstance(ce_loss_fn, torch.nn.CrossEntropyLoss):
            print("Validation Perplexity: %f"%(np.exp(valid_loss)))
        else:
            print("Validation loss: %f"%valid_loss)
    
    print("best checkpoint is %d"%best_ckpt)
    _, model, _, _ = load_ckpt(best_ckpt_path, False)
    print("testing...")
    results = batch_evaluate(model, test_iter,train_iter.dataset.src_lang, train_iter.dataset.trg_lang,device=args.device,beam_size=5)
    print("computing BLEU score...")
    bleu_score = metric(results, metric_type="bleu")
    print("BLEU: %f"%bleu_score)
    

if __name__=="__main__":
    args= parse_arguments()
    main(args)
                       
