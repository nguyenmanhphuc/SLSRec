from .lru import *
from .slsrec import SLSRecTrainer
from .recblr import RecBLRTrainer


TRAINERS = {
    'lru': LRUTrainer,
    'slsrec': SLSRecTrainer,
    'recblr': RecBLRTrainer,
}


def trainer_factory(args, model, train, val, test, export_root, use_wandb):
    return TRAINERS[args.model_code](args, model, train, val, test, export_root, use_wandb)
