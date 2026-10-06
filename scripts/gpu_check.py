"""Will this configuration fit and how long will it take -- on THIS machine?

Runs a few real training steps at the paper's batch size (128 x 200 on ML-1M)
and reports peak GPU memory, seconds per step, and an extrapolated run time.
Takes the same flags as train.py, so check exactly the command you plan to run.

    python scripts/gpu_check.py --dataset_code=ml-1m
    python scripts/gpu_check.py --dataset_code=ml-1m --uncertainty_loss=exact
    python scripts/gpu_check.py --dataset_code=ml-1m --gate_mode=token --uncertainty_loss=exact

Without a GPU it still runs (on CPU) and prints the activation-memory estimate,
so the script itself can be checked anywhere.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                          # noqa: E402

from datasets import DATASETS                          # noqa: F401,E402 (import order)
import config as cfg                                   # noqa: E402

STEPS = int(os.environ.get('GPU_CHECK_STEPS', 8))
WARMUP = 2
BATCHES_PER_EPOCH = {'ml-1m': 56}
NUM_ITEMS = {'ml-1m': 3416}
MB = 1024 ** 2


def resolve(argv):
    flags = list(argv)
    if not any(f.startswith('--dataset_code') for f in flags):
        flags.append('--dataset_code=ml-1m')
    if not any(f.startswith('--model_code') for f in flags):
        flags.append('--model_code=slsrec')
    a = cfg.parser.parse_args(flags)
    cfg.set_template(a)
    a.num_items = NUM_ITEMS.get(a.dataset_code, 3416)
    a.num_users = 6040
    return a


def saved_activation_bytes(a, model, seqs, labels):
    """Bytes autograd keeps for backward -- device independent."""
    from trainer.slsrec import slsrec_losses
    seen = {}

    def pack(t):
        s = t.untyped_storage()
        seen[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        slsrec_losses(a, model, seqs, labels, training=True)
    return sum(seen.values())


def main():
    a = resolve(sys.argv[1:])
    from model import model_factory
    from trainer.slsrec import slsrec_losses

    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    print(f'torch {torch.__version__} | CUDA build {torch.version.cuda} | GPU available: {cuda}')
    if cuda:
        p = torch.cuda.get_device_properties(0)
        total = p.total_memory
        print(f'GPU: {p.name} | {total / MB:.0f} MB | compute capability {p.major}.{p.minor}')
        free, _ = torch.cuda.mem_get_info()
        print(f'free right now: {free / MB:.0f} MB (other apps / the display may hold the rest)')
    else:
        print('No CUDA device -- timing below is CPU.  Install a CUDA build of torch to use the GPU.')

    B, L, V = a.train_batch_size, a.bert_max_len, a.num_items + 1
    print(f'\nconfig: batch {B} x len {L} | items {V - 1} | gate {a.gate_mode}/{a.fusion_mode} | '
          f'L_Unc {a.uncertainty_loss} | perturbation {a.use_perturbation} | uncertainty {a.use_uncertainty}')

    torch.manual_seed(a.seed)
    model = model_factory(a).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, eps=a.adam_epsilon)
    seqs = torch.randint(1, V, (B, L), device=device)
    labels = torch.randint(1, V, (B, L), device=device)

    est = saved_activation_bytes(a, model, seqs, labels)
    logits = B * L * V * 4
    print(f'\nactivations saved for backward: {est / MB:.0f} MB '
          f'(one [B,L,|I|] logits tensor = {logits / MB:.0f} MB)')
    model.zero_grad(set_to_none=True)
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def step():
        opt.zero_grad(set_to_none=True)
        losses = slsrec_losses(a, model, seqs, labels, training=True)
        losses['L_total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
        opt.step()
        return losses['L_total'].item()

    try:
        for _ in range(WARMUP):
            step()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            loss = step()
        if cuda:
            torch.cuda.synchronize()
        per_step = (time.perf_counter() - t0) / STEPS
    except torch.cuda.OutOfMemoryError:
        print('\n*** OUT OF MEMORY at the paper batch size (128). ***')
        print('    Do not shrink the batch -- that changes the experiment.  See the notes in')
        print('    the reply / reproduction_report for memory-saving options.')
        return 2

    # one evaluation batch, the way calculate_metrics does it
    model.eval()
    ev = torch.randint(1, V, (a.val_batch_size, L), device=device)
    with torch.no_grad():
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sc = model(ev)[0][:, -1, :]
        sc.scatter_(1, ev, -1e9)
        sc.argsort(dim=1)
        if cuda:
            torch.cuda.synchronize()
        per_eval = time.perf_counter() - t0

    per_epoch_batches = BATCHES_PER_EPOCH.get(a.dataset_code, 56)
    val_batches = 24
    vals_per_epoch = (1 / a.val_epochs) if a.val_strategy == 'epoch' \
        else per_epoch_batches / a.val_iterations
    epoch_s = per_step * per_epoch_batches + vals_per_epoch * val_batches * per_eval

    print(f'\nloss finite: {torch.isfinite(torch.tensor(loss)).item()}')
    print(f'train step:  {per_step * 1000:7.1f} ms')
    print(f'eval batch:  {per_eval * 1000:7.1f} ms')
    print(f'per epoch:   {epoch_s:7.1f} s   ({per_epoch_batches} steps + amortised validation)')
    for n in (200, 400, 1000):
        print(f'  {n:>4} epochs  ~ {epoch_s * n / 3600:5.1f} h')

    if cuda:
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_res = torch.cuda.max_memory_reserved()
        frac = peak_res / total
        print(f'\npeak allocated: {peak_alloc / MB:.0f} MB | peak reserved: {peak_res / MB:.0f} MB '
              f'| {100 * frac:.0f}% of {total / MB:.0f} MB')
        if frac > 0.92:
            verdict = 'TOO TIGHT -- expect OOM or heavy slowdown in a long run.'
        elif frac > 0.80:
            verdict = 'TIGHT -- runs, but close other GPU apps and run one job at a time.'
        else:
            verdict = 'OK -- fits with headroom.'
        print(f'verdict: {verdict}')
        fits = int(0.9 * total // peak_res) if peak_res else 0
        print(f'concurrent copies that fit: {max(fits, 0)}  (sweep scripts: use -Jobs / JOBS <= this)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
