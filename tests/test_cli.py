"""train.py must refuse flags it does not know.

config.py parses with parse_known_args so it can be imported from pytest and
from scripts that carry their own flags.  For training that is dangerous: a
flag newer than the checkout (e.g. --unc_dropout_form on an older commit) used
to be dropped with a one-line warning, silently re-running a different
experiment.  These tests pin the guard in train.py.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*flags):
    return subprocess.run([sys.executable, 'train.py', *flags], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)


def test_unknown_flag_aborts_before_training():
    r = _run('--dataset_code=ml-1m', '--model_code=slsrec', '--no_such_flag=1')
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert 'unrecognised argument' in out and '--no_such_flag=1' in out
    assert 'Total parameters' not in out            # never reached the trainer


def test_misspelt_flag_aborts():
    r = _run('--dataset_code=ml-1m', '--model_code=slsrec', '--unc_dropout_from=eq20')
    assert r.returncode != 0
    assert '--unc_dropout_from=eq20' in r.stdout + r.stderr


def test_known_flags_parse_cleanly():
    code = ("import sys; sys.argv = ['train.py', '--dataset_code=ml-1m', '--model_code=slsrec', "
            "'--unc_dropout_form=eq20', '--uncertainty_loss=exact', '--num_workers=2', "
            "'--exp_name=x']; import config; print(repr(config._unknown_args))")
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == '[]'
