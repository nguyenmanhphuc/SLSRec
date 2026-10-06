Import shims for running the *unmodified* `_ref_LRURec/` source.

`_ref_LRURec/train.py` imports two packages that the reproduction does not use:

* `wandb` -- imported at module level but never called (`--use_wandb` is False
  by default, and `trainer/base.py` imports it lazily inside the `if` branch).
* `pytorch_lightning` -- only for `seed_everything`.

Installing pytorch-lightning on a pod risks pip resolving a different torch, so
these stubs stand in instead.  They are put on `PYTHONPATH` by
`scripts/verify_lrurec.sh`, which prefers the real packages whenever they are
already installed, and they never touch a file under `_ref_LRURec/`.
