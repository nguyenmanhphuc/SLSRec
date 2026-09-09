from .lru import *
from .bdlru import BDLRU, CausalDepthwiseConv1d, parallel_scan, sequential_scan, hillis_steele_scan
from .slsrec import SLSRec, SLSRecModel


MODELS = {
    'lru': LRU,
    'slsrec': SLSRec,
}


def model_factory(args):
    return MODELS[args.model_code](args)
