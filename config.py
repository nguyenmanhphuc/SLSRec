import torch
import numpy as np
import argparse
import random
import sys

# NOTE: these constants are defined *before* the package imports below so that
# `import config` works standalone -- datasets/base.py imports
# RAW_DATASET_ROOT_FOLDER from here, which would otherwise be a circular import.
RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'

STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'

PROJECT_NAME = 'recsys'

from datasets import *
from model import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('boolean value expected, got %r' % v)


def set_template(args):
    user_bert_dropout = args.bert_dropout  # None unless explicitly passed
    args.min_uc = 5
    args.min_sc = 5
    args.split = 'leave_one_out'
    
    if args.dataset_code == None:
        print('******************** Dataset Selection ********************')
        dataset_code = {'1': 'ml-1m', 'b': 'beauty', 's': 'sports', 't': 'steam', 'v': 'video', 'x': 'xlong'}
        args.dataset_code = dataset_code[input('Input 1 for ml-1m, b for beauty, s for sports, t for steam, v for video and x for xlong: ')]
    
    # Per-dataset defaults, but never override a value the user passed explicitly
    # (upstream LRURec clobbers --bert_max_len / --val_iterations here).
    if args.dataset_code == 'ml-1m':
        max_len, val_iters = 200, 500
    elif args.dataset_code in ('steam', 'xlong'):
        max_len, val_iters = (1000, 2000) if args.dataset_code == 'xlong' else (50, 2000)
    else:  # beauty, sports, video, yelp
        max_len, val_iters = 50, 1000
    if args.bert_max_len is None: args.bert_max_len = max_len
    if args.val_iterations is None: args.val_iterations = val_iters
    
    batch = 32 if args.dataset_code == 'xlong' else 128
    args.train_batch_size = batch
    args.val_batch_size = batch * 2
    args.test_batch_size = batch * 2

    if args.model_code is None: args.model_code = 'lru'

    # Training schedule.  SLSRec follows its paper (Sec. IV-A4): "training is
    # limited to a maximum of 1000 epochs.  Validation is conducted every 5
    # epochs, and ... no improvement in the validation Recall@10 metric for 10
    # consecutive validation epochs".  The same paragraph applies it to the
    # baselines it re-ran in the LRURec repository, so RecBLR gets it too.  The
    # LRURec baseline keeps LRURec's own protocol (500 epochs, validate every
    # val_iterations steps).  Explicitly passed flags always win.
    paper_schedule = args.model_code in ('slsrec', 'recblr')
    if args.num_epochs is None: args.num_epochs = 1000 if paper_schedule else 500
    if args.val_strategy is None: args.val_strategy = 'epoch' if paper_schedule else 'iteration'
    if args.val_epochs is None: args.val_epochs = 5 if paper_schedule else 1
    if torch.cuda.is_available(): args.device = 'cuda'
    else: args.device = 'cpu'
    args.optimizer = 'AdamW'
    if args.lr is None: args.lr = 0.001
    if args.weight_decay is None: args.weight_decay = 0.01
    if args.bert_dropout is None: args.bert_dropout = 0.2
    if args.bert_attn_dropout is None: args.bert_attn_dropout = 0.2
    if args.bert_mask_prob is None: args.bert_mask_prob = 0.2
    
    args.enable_lr_schedule = False
    args.decay_step = 10000
    args.gamma = 0.1
    args.enable_lr_warmup = False
    args.warmup_steps = 100

    args.metric_ks = [1, 5, 10, 20, 50]
    args.best_metric = 'Recall@10'
    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None

    set_slsrec_template(args, user_bert_dropout)


# Per-dataset dropout optima, read off SLSRec Figs. 6 and 7 (see the docstring
# of set_slsrec_template).  xlong is not in the paper; it keeps the 0.2 default.
SLSREC_DROPOUT = {
    'ml-1m': 0.2,
    'beauty': 0.75,
    'video': 0.6,
    'sports': 0.75,
    'steam': 0.2,
}


def set_slsrec_template(args, user_bert_dropout=None):
    """Per-dataset SLSRec hyperparameters from the SLSRec paper.

    Dropout: read off the per-dataset optima in Figs. 6 and 7 (Recall@20 and
    NDCG@20 vs dropout).  The prose in Sec. IV-E1 only names ML-1M (0.2) and
    Sports (0.75); the figures give all five, and each optimum reproduces the
    corresponding Table II entry:

        ML-1M 0.2 | Beauty 0.75 | Video 0.6 | Sports 0.75 | Steam 0.2

    Note Video: Sec. IV-A4 groups it with ML-1M and Steam by *search grid*
    ({0, 0.2, 0.4, 0.6, 0.8}), which reads as "dense -> 0.2", but its curve
    actually peaks at 0.6.  Reading 0.2 off the grid grouping gives ~0.158
    Recall@20 in Fig. 6, while Table II reports 0.17353 -- the 0.6 point.

    Perturbation rate theta = 0.1 for all datasets (Sec. IV-E2: "we use a
    default perturbation rate of 0.1 in our experimental setup"; every Table II
    entry matches the theta = 0.1 column of the theta grid).
    Uncertainty rate alpha = 0.01 for Beauty, 0.3 for every other dataset
    (Sec. IV-E3, stated explicitly).
    """
    if args.model_code != 'slsrec':
        return
    if args.uncertainty_alpha is None:
        args.uncertainty_alpha = 0.01 if args.dataset_code == 'beauty' else 0.3
    if args.slsrec_dropout is not None:
        args.bert_dropout = args.slsrec_dropout
    elif user_bert_dropout is None:
        args.bert_dropout = SLSREC_DROPOUT.get(args.dataset_code, 0.2)


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default=None)
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=2)
parser.add_argument('--min_sc', type=int, default=1)
parser.add_argument('--split', type=str, default='leave_one_out')
parser.add_argument('--seed', type=int, default=42)

################
# Dataloader
################
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--sliding_window_size', type=float, default=1)
parser.add_argument('--negative_sample_size', type=int, default=100)
parser.add_argument('--xlong_negative_sample_size', type=int, default=10000)

################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
# optimizer & lr#
parser.add_argument('--num_epochs', type=int, default=None)  # slsrec: 1000 (paper), lru: 500
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'])
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=100)

# evaluation #
parser.add_argument('--val_strategy', type=str, default=None, choices=['epoch', 'iteration'])  # slsrec: epoch, lru: iteration
parser.add_argument('--val_iterations', type=int, default=None)  # only for iteration val_strategy
parser.add_argument('--val_epochs', type=int, default=None)  # only for epoch val_strategy; slsrec: 5, lru: 1
                                                          # (SLSRec paper validates every 5 epochs)
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=10)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20, 50])
parser.add_argument('--best_metric', type=str, default='Recall@10')
parser.add_argument('--use_wandb', type=bool, default=False)

################
# Model
################
parser.add_argument('--model_code', type=str, default=None, choices=[None, 'lru', 'slsrec', 'recblr'])
# BERT specs, used for other models as well #
parser.add_argument('--bert_max_len', type=int, default=None)
parser.add_argument('--bert_hidden_units', type=int, default=64)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=2)
parser.add_argument('--bert_head_size', type=int, default=32)
parser.add_argument('--bert_dropout', type=float, default=None)
parser.add_argument('--bert_attn_dropout', type=float, default=None)
parser.add_argument('--bert_mask_prob', type=float, default=None)

################
# SLSRec
################
# architecture #
parser.add_argument('--slsrec_expand', type=int, default=2,
                    help='latent width d = expand * D (paper: d = 2D; also RecBLR expand)')
parser.add_argument('--slsrec_d_conv', type=int, default=4,
                    help='causal depth-wise convolution kernel size (RecBLR: 4; also RecBLR d_conv)')
parser.add_argument('--slsrec_dropout', type=float, default=None,
                    help='override the per-dataset SLSRec dropout rate')
parser.add_argument('--sglru_dropout', type=float, default=None,
                    help='dropout on the SG-LRU output, eq. (7) only; default = the shared rate. '
                         'LRURec wires this position to --bert_attn_dropout, not --bert_dropout')
parser.add_argument('--scan_impl', type=str, default='torch',
                    choices=['torch', 'sequential', 'triton'])
parser.add_argument('--use_item_bias', type=str2bool, default=False,
                    help='add a per-item bias to the logits (LRURec does; eq. (15) does not)')
parser.add_argument('--pffn_bias', type=str2bool, default=False,
                    help='biases on W6/W7/W8 of every PFFN (eq. (13) and (20) print none; '
                         'the LRURec and RecBLR FFNs have them)')

# self-attention gate, eq. (3)/(8) #
parser.add_argument('--use_sam_gate', type=str2bool, default=True,
                    help='include the SAM(L1) term in the gate G')
parser.add_argument('--sam_conv', type=str2bool, default=True,
                    help='include the Conv1-D1(L1) term in the gate G')
parser.add_argument('--sam_conv_activation', type=str, default='none', choices=['none', 'silu'])
parser.add_argument('--sam_num_heads', type=int, default=1,
                    help='eq. (8) is single-head')
parser.add_argument('--sam_attn_dropout', type=float, default=0.0)

# BD-LRU, eq. (9)-(12) #
parser.add_argument('--use_bdlru', type=str2bool, default=True)
parser.add_argument('--bdlru_c', type=float, default=8.0, help='scalar c in eq. (11)')
parser.add_argument('--bdlru_r_min', type=float, default=0.9)
parser.add_argument('--bdlru_r_max', type=float, default=0.999)
parser.add_argument('--bdlru_lambda_init', type=str, default='linspace',
                    choices=['linspace', 'uniform'])
parser.add_argument('--bdlru_conv_activation', type=str, default='silu', choices=['none', 'silu'])
parser.add_argument('--exclude_lambda_from_decay', type=str2bool, default=False,
                    help='exclude the BD-LRU lambda from weight decay (RecBLR uses wd=0; '
                         'LRURec decays its recurrence parameters, which is the default here)')

# perturbation injection, eq. (17)/(18) #
parser.add_argument('--use_perturbation', type=str2bool, default=True)
parser.add_argument('--perturbation_probability', type=float, default=0.1, help='theta in eq. (17)')
parser.add_argument('--perturbation_low', type=float, default=0.0)
parser.add_argument('--perturbation_high', type=float, default=1.0)
parser.add_argument('--perturbation_weight', type=float, default=1.0,
                    help='coefficient of L_Per in eq. (23); the paper uses 1')
# alternative readings of eq. (17) -- defaults are the literal one
parser.add_argument('--perturbation_site', type=str, default='bdlru',
                    choices=['bdlru', 'bdlru_conv', 'layer', 'embedding'],
                    help="where x_t is replaced.  bdlru = the BD-LRU input x_t of eq. (9)-(12), as "
                         "Sec. III-C1 writes it; bdlru_conv = L2, before Conv1-D2; layer = the input "
                         "X^(l-1) of every SG-LRU layer (gate branch, BD-LRU branch and residual); "
                         "embedding = X^(0) once, before the first layer")
parser.add_argument('--perturbation_noise', type=str, default='uniform', choices=['uniform', 'shuffle'],
                    help='uniform = U(low, high) as in eq. (17); shuffle = the vector at a random real '
                         'position of another sequence in the batch (in-distribution noise)')
parser.add_argument('--perturbation_granularity', type=str, default='position',
                    choices=['position', 'element'],
                    help='position = one draw per step replaces the whole vector (eq. 17); '
                         'element = every component is replaced independently')
parser.add_argument('--perturbation_gates', type=str, default='perturbed', choices=['perturbed', 'clean'],
                    help='site=bdlru only.  perturbed = r_t, i_t of eq. (9)/(10) see the replaced x_t '
                         '(literal "replace x_t in (9)-(12)"); clean = the gates see the clean x_t and '
                         'only the input term of eq. (12) is replaced, so the noise cannot be gated out')

# uncertainty scaling, eq. (19)-(24) #
parser.add_argument('--use_uncertainty', type=str2bool, default=True)
parser.add_argument('--uncertainty_alpha', type=float, default=None, help='alpha in eq. (23)')
parser.add_argument('--uncertainty_beta', type=float, default=0.0,
                    help='beta in eq. (22); the paper sets it to 0')
parser.add_argument('--uncertainty_loss', type=str, default='approx', choices=['approx', 'exact'],
                    help='approx = eq. (22) (used by the paper for eq. (24)); exact = eq. (19)/(21)')
parser.add_argument('--unc_dropout_form', type=str, default='fig2', choices=['fig2', 'eq20'],
                    help="dropout placement in the uncertainty estimator.  fig2 = "
                         "LN(Dropout(PFFN(xc)) + xc), as drawn in Fig. 2 and as every other "
                         "block (Fig. 1, eq. 7, eq. 14); eq20 = LN(Dropout(PFFN(xc) + xc)), as "
                         "eq. (20) is printed.  Identical at eval time.")
parser.add_argument('--uncertainty_logvar_clamp', type=float, default=8.0,
                    help='numerical safety clamp on Log(sigma^2); 0 disables')
# alternative readings of eq. (19)-(24) -- defaults are the ones used so far
parser.add_argument('--uncertainty_granularity', type=str, default='position',
                    choices=['position', 'sequence', 'global'],
                    help="position = one sigma per prefix, from its last item and its target "
                         "(per-position training); sequence = one sigma_u per sequence from its last "
                         "item and ground truth, as Sec. III-C2 words it, shared by all its positions; "
                         "global = one learned sigma for every sequence (the [45] approach)")
parser.add_argument('--uncertainty_weight_norm', type=str, default='none', choices=['none', 'batch'],
                    help='batch = divide 1/sigma^2 by its batch mean, so U can move weight between '
                         'samples but cannot switch itself off.  A control, not in the paper')

################
# Adaptive gating -- RESEARCH EXTENSION, not part of the SLSRec paper.
# Defaults reproduce the paper exactly (gate_mode=sum, fusion_mode=mul).
################
# axis A: how G is composed from SAM and Conv1-D1  (replaces eq. 3)
parser.add_argument('--gate_mode', type=str, default='sum',
                    choices=['sum', 'scalar', 'channel', 'token', 'context'],
                    help="sum = original eq.(3); scalar/channel = learned but static; "
                         "token/context = input-dependent adaptive gate")
# axis B: how strongly G gates H  (replaces eq. 6)
parser.add_argument('--fusion_mode', type=str, default='mul', choices=['mul', 'adaptive'],
                    help="mul = original eq.(6) G*H; adaptive = (1 + a*(G-1))*H")
parser.add_argument('--fusion_source', type=str, default='token', choices=['token', 'context'])
parser.add_argument('--gate_granularity', type=str, default='channel', choices=['scalar', 'channel'],
                    help='one coefficient per position, or per position and channel')
parser.add_argument('--gate_hidden', type=int, default=0,
                    help='>0 turns the gate projection into a bottleneck MLP of this width')
parser.add_argument('--gate_neutral_init', type=str2bool, default=True,
                    help='start every gate at its neutral point so step 0 equals original SLSRec')

# bookkeeping #
parser.add_argument('--smoke_iters', type=int, default=10,
                    help='scripts/smoke_ablations.py: training steps per ablation')
parser.add_argument('--smoke_val_batches', type=int, default=4,
                    help='scripts/smoke_ablations.py: validation batches per ablation')
parser.add_argument('--exp_name', type=str, default=None,
                    help='experiment folder name under experiments/<model_code>/')

# parse_known_args (rather than parse_args) so that importing this module from
# pytest -- which owns sys.argv -- does not abort.
args, _unknown_args = parser.parse_known_args()
if _unknown_args and 'pytest' not in sys.modules:
    print('[config] ignoring unrecognised arguments:', _unknown_args)
