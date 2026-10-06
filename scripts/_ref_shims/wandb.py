"""Stub for `import wandb` in _ref_LRURec/train.py.

Only used when wandb is not installed.  LRURec imports the module at the top of
train.py but never touches it unless --use_wandb is passed, so raising on any
attribute access is both safe and loud if that assumption ever breaks.
"""


def __getattr__(name):
    raise RuntimeError(
        'the wandb stub was used (attribute %r): install the real wandb, or '
        'run without --use_wandb' % name)
