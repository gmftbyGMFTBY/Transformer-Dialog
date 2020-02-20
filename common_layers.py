'''
@Description: 
@Author: bairongz (baiyuu.cs@gmail.com)
@Date: 2019-12-13 16:47:37
@LastEditTime : 2019-12-21 14:34:07
'''
import torch
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class Noam(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps, d_model):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super(Noam, self).__init__(optimizer)
        # self.i_step = 0

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = (self.d_model** -0.5) * min([
            last_epoch ** (-0.5), 
            last_epoch * self.warmup_steps ** (-1.5),
            ])

        # return [base_lr  * scale for base_lr in self.base_lrs]
        return [base_lr / base_lr * scale for base_lr in self.base_lrs]


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps,
            ])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class ACT_basic(nn.Module):
    def __init__(self, d_model, pos_embed, time_embed, epslion=1e-1, act_type="t2t"):
        super(ACT_basic, self).__init__()
        self.proj = nn.Linear(d_model,1)  
        self.proj.bias.data.fill_(1) 
        self.threshold = 1. - epslion
        self.pos_embed = pos_embed
        self.time_embed = time_embed
        self.d_model = d_model
        self.act_type = act_type
        if act_type =="rnn":
            self.rnn = nn.GRU(d_model, d_model//2, 2, bidirectional=True)
            self.rnn_dec = nn.GRU(d_model, d_model, 2,)

    def forward(self, x, x_pos, max_hops, inputs, state_fn, device="cpu", is_decoder=False):
        # adapted from tensor2tensor
        if self.act_type == "t2t":
            L, B, _ = x.shape
            halting_probability = torch.zeros(L, B).to(device)
            remainders = torch.zeros(L, B).to(device)
            n_updates = torch.zeros(L, B).to(device)
            previous_state = torch.zeros_like(x).to(device)

            state = x
            for i_time in range(1, max_hops+1):
                if not (halting_probability<self.threshold).any():
                    break
                
                state_time = torch.tensor([i_time]*B).to(device).squeeze(-1)
                state = state + self.pos_embed(x_pos) + self.time_embed(state_time)
                
                p = torch.sigmoid(self.proj(state)).squeeze(-1)
                
                still_running_mask = (halting_probability< 1.).float()
                new_halted_mask = ((halting_probability + p * still_running_mask) > self.threshold).float() * still_running_mask
                still_running_mask = ((halting_probability + p * still_running_mask) <= self.threshold).float() * still_running_mask
                halting_probability = halting_probability + p * still_running_mask
                remainders = remainders + new_halted_mask * (1 - halting_probability)
                halting_probability = halting_probability + new_halted_mask * remainders
                n_updates = n_updates + still_running_mask + new_halted_mask
                update_weights = p * still_running_mask + new_halted_mask * remainders
                if is_decoder:
                    state = state_fn(state,inputs["memory"] ,tgt_mask=inputs["trg_mask"], memory_mask=inputs["memory_mask"], tgt_key_padding_mask=inputs["trg_key_padding_mask"], memory_key_padding_mask=inputs["memory_key_padding_mask"])
                else:
                    state = state_fn(state, src_mask=inputs["src_mask"], src_key_padding_mask=inputs["src_key_padding_mask"])
                    
                previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1. - update_weights.unsqueeze(-1))))

        elif self.act_type=="rnn":
            L, B, _ = x.shape

            halting_probability_sum = torch.zeros(L, B).to(device)
            remainders = torch.zeros(L, B).to(device)
            n_updates = torch.zeros(L, B).to(device)
            previous_state = torch.zeros_like(x).to(device)

            state = x
            for i_time in range(1, max_hops+1):
                if not (halting_probability_sum<self.threshold).any():
                    break
                
                state_time = torch.tensor([i_time]*B).to(device).squeeze(-1)
                state = state + self.pos_embed(x_pos) + self.time_embed(state_time)

                # probability for halting at current time step, (L, B,  1)
                # p = torch.sigmoid(self.proj(state)).squeeze(-1)
                if is_decoder:
                    p = torch.sigmoid(self.proj(self.rnn_dec(state)[0])).squeeze(-1)
                else:
                    p = torch.sigmoid(self.proj(self.rnn(state)[0])).squeeze(-1)
                
                still_running_mask = (halting_probability_sum< 1.).float()

                new_halted_mask = ((halting_probability_sum + p * still_running_mask) > self.threshold).float() * still_running_mask
                still_running_mask = ((halting_probability_sum + p * still_running_mask) <= self.threshold).float() * still_running_mask
                halting_probability_sum +=  p * still_running_mask
                
                remainders += new_halted_mask * (1 - halting_probability_sum)
                halting_probability_sum = halting_probability_sum + new_halted_mask * remainders
                n_updates += still_running_mask + new_halted_mask
                update_weights = p * still_running_mask + new_halted_mask * remainders
                
                if is_decoder:
                    state = state_fn(state,inputs["memory"] ,tgt_mask=inputs["trg_mask"], memory_mask=inputs["memory_mask"], tgt_key_padding_mask=inputs["trg_key_padding_mask"], memory_key_padding_mask=inputs["memory_key_padding_mask"])
                else:
                    state = state_fn(state, src_mask=inputs["src_mask"], src_key_padding_mask=inputs["src_key_padding_mask"])
                
                previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1. - update_weights.unsqueeze(-1))))
        
        else:
            raise ValueError("Unknow ACT type: %s"%self.act_type)

        # there is no trg mask in testing time
        if is_decoder and inputs["trg_key_padding_mask"]:
            mask = 1. - inputs["trg_key_padding_mask"].permute(1, 0).float()
        elif not is_decoder:
            mask = 1. - inputs["src_key_padding_mask"].permute(1, 0).float()
        else:
            return previous_state, (remainders, n_updates)

        return previous_state, (remainders*mask, n_updates*mask)

class UniversalTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, n_layers, norm, pos_embed, time_embed, act=None):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.n_layers= n_layers
        self.norm = norm
        self.pos_embed = pos_embed
        self.time_embed = time_embed
        self.act = act
        self.remainders=0.
        self.n_updates=0.
        
    def forward(self, src, src_pos, src_mask=None, src_key_padding_mask=None):
        L,B,H = src.shape
        device = src.device
        outputs=[]
        if self.act:
            src_memory, (self.remainders,self.n_updates) = self.act(src, src_pos, self.n_layers, {"src_mask":src_mask,"src_key_padding_mask":src_key_padding_mask}, self.encoder_layer, device=device) 
            src_memory = self.norm(src_memory)
            
            return src_memory
        else:
            src_memory= src
            for i_time in range(1, 1+self.n_layers):
                src_time = torch.tensor([i_time]*B).to(src.device)
                src_memory = src_memory + self.pos_embed(src_pos) + self.time_embed(src_time)
                src_memory = self.encoder_layer(src_memory, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
                src_memory = self.norm(src_memory)
                outputs.append(src_memory)
        
            return outputs[-1]
                      
class UniversalTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, n_layers, norm, pos_embed, time_embed, act=None):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.n_layers= n_layers
        self.norm = norm
        self.pos_embed = pos_embed
        self.time_embed = time_embed
        self.act = act
        self.remainders=0.
        self.n_updates=0.
    
    def forward(self, trg, trg_pos, memory, trg_mask=None, memory_mask=None, trg_key_padding_mask=None, memory_key_padding_mask=None):
        L,B,H = trg.shape
        device = trg.device

        outputs=[]
        if self.act:
            src_memory, (self.remainders, self.n_updates) = self.act(trg,trg_pos,self.n_layers,{"memory":memory,"trg_mask":trg_mask, "memory_mask":memory_mask, "trg_key_padding_mask":trg_key_padding_mask, "memory_key_padding_mask":memory_key_padding_mask},self.decoder_layer, is_decoder=True, device=device)    
            src_memory = self.norm(src_memory)
        
            return src_memory
        else:
            trg_embed = trg
            for i_time in range(1, 1+self.n_layers):
                trg_time = torch.tensor([i_time]*B).to(trg.device)
                trg_embed = trg_embed + self.pos_embed(trg_pos) + self.time_embed(trg_time)
                trg_embed = self.decoder_layer(trg_embed, memory, tgt_mask=trg_mask, memory_mask=memory_mask, tgt_key_padding_mask=trg_key_padding_mask,memory_key_padding_mask=memory_key_padding_mask)
                trg_embed = self.norm(trg_embed)
                outputs.append(trg_embed)
            
            return outputs[-1]

class CEWithLabelSmoothing(nn.Module):
    def __init__(self, vocab_size, label_smoothing=.1, ignore_index=-1,reduction="word_mean"):
        """Cross Entropy Loss with Label Smoothing
        
        Arguments:
            vocab_size {int} -- # of vocabulary in the target language
        
        Keyword Arguments:
            label_smoothing {float} -- label smoothing factor (default: {.1})
            ignore_index {int} -- index need to ignore when calculate the loss (default: {-1})
            reduction {str} -- value in {"word_mean", "sum"}, "word mean": compute word level average loss, "sum":total loss (default: {"word_mean"}) 
        """
        super(CEWithLabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.ignore_index = ignore_index
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.vocab_size = vocab_size
        self._true_dist = None
        self._reduction = reduction


        
    def forward(self, logits, target):
        assert logits.size(1) == self.vocab_size, "size mismatch! %d!=%d"%(logits.size(1),self.vocab_size)

        true_dist = logits.clone()
        true_dist.fill_(self.label_smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.ignore_index] = 0.

        mask = (target !=self.ignore_index).float().unsqueeze(1)
        true_dist = true_dist*mask

        loss = self.criterion(self.log_softmax(logits), true_dist)
        
        n_words = torch.sum(mask)
        
        # save some data for debugging
        self._true_dist = true_dist
        self._kl = loss
        self._n_words = n_words

        if self._reduction =="word_mean":
            return loss / n_words
        elif self._reduction=="sum":
            return loss
        else:
            raise ValueError

class BaseGenerator(nn.Module):
    def __init__(self, d_model, n_vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        logits = self.proj(x)
        return logits

class TwoStageGenerator(nn.Module):
    def __init__(self,d_embed, d_model, n_vocab):
        super().__init__()
        self.hidden2embed = nn.Linear(d_model, d_embed)
        self.embed2vocab = nn.Linear(d_embed, n_vocab)

    def forward(self, x):
        x = torch.tanh(self.hidden2embed(x))
        logits = self.embed2vocab(x)
        return logits
        
        
