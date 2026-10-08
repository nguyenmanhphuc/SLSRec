from .lru import *
from .bdlru import BDLRU, CausalDepthwiseConv1d, parallel_scan, sequential_scan, hillis_steele_scan
from .slsrec import SLSRec, SLSRecModel
from .recblr import RecBLR


MODELS = {
    'lru': LRU,
    'slsrec': SLSRec,
    'recblr': RecBLR,
}


def model_factory(args):
    return MODELS[args.model_code](args)
