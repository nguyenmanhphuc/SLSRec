import torch
import argparse

from datasets import DATASETS
from config import *
from model import *
from dataloader import *
from trainer import *
from trainer.utils import seed_everything


def train(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        if getattr(args, 'exp_name', None):
            export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.exp_name
        else:
            export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code + \
                '_' + str(args.weight_decay) + '_' + str(args.bert_dropout) + '_' + str(args.bert_attn_dropout)

    train, val, test = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train, val, test, export_root, args.use_wandb)
    trainer.train()
    return trainer.test()


if __name__ == "__main__":
    # config.py parses with parse_known_args so it stays importable from pytest
    # and from scripts with their own flags -- but for training, an unknown flag
    # must not be dropped with a one-line warning: a flag this checkout does not
    # know (e.g. a newer option run against older code) would silently rerun a
    # different experiment.
    from config import _unknown_args
    if _unknown_args:
        raise SystemExit('[train.py] unrecognised argument(s): {} -- refusing to train with '
                         'flags that would be silently ignored. Check the spelling, or update '
                         'the code if the flag is newer than this checkout.'
                         .format(' '.join(_unknown_args)))
    set_template(args)
    train(args)
