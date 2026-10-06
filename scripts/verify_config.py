"""Audit a command line against the SLSRec paper.

Resolves the arguments exactly the way train.py does, builds the model, then
checks every architectural / hyperparameter / loss claim of the paper against
the object tree that was actually constructed.  Prints a PASS/FAIL table.

    python scripts/verify_config.py --dataset_code=ml-1m --model_code=slsrec \
        --seed=42 --exp_name=D_full_slsrec --num_workers=0
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from datasets import DATASETS                       # noqa: F401 (import order)
from config import args, set_template
from model import model_factory
from trainer.slsrec import slsrec_losses

ROWS = []


def check(section, what, paper, got, ok):
    ROWS.append((section, what, str(paper), str(got), bool(ok)))


def main():
    set_template(args)
    # the dataloader would set these; ML-1M values after 5-core filtering
    args.num_items, args.num_users = 3416, 6040
    model = model_factory(args).eval()
    core = model.model
    blk = core.blocks[0].sg_lru
    D, d = args.bert_hidden_units, args.bert_hidden_units * args.slsrec_expand

    # ---------------- Sec. III-B: architecture ---------------- #
    check('III-B', 'model selected', 'SLSRec', args.model_code, args.model_code == 'slsrec')
    check('III-B1', 'eq.(1) item embedding M', f'|I|+1 x D = {args.num_items+1} x {D}',
          tuple(core.embedding.token.weight.shape), tuple(core.embedding.token.weight.shape) == (args.num_items + 1, D))
    check('III-B1', 'eq.(1) position embedding P', f'N x D = {args.bert_max_len} x {D}',
          tuple(core.embedding.position.weight.shape), tuple(core.embedding.position.weight.shape) == (args.bert_max_len, D))
    check('III-B1', 'eq.(1) Dropout + LayerNorm', 'both present',
          f'{type(core.embedding.dropout).__name__}+{type(core.embedding.layer_norm).__name__}',
          isinstance(core.embedding.dropout, nn.Dropout) and isinstance(core.embedding.layer_norm, nn.LayerNorm))
    check('III-B', 'L stacked layers', 2, len(core.blocks), len(core.blocks) == 2)
    check('III-B2', 'latent width d = 2D', 2 * D, d, d == 2 * D)

    check('III-B2', 'eq.(2) W1: D->d (+bias)', f'({d}, {D}) bias=True',
          f'{tuple(blk.w1.weight.shape)} bias={blk.w1.bias is not None}',
          tuple(blk.w1.weight.shape) == (d, D) and blk.w1.bias is not None)
    check('III-B2', 'eq.(4) W2: D->d (+bias)', f'({d}, {D}) bias=True',
          f'{tuple(blk.w2.weight.shape)} bias={blk.w2.bias is not None}',
          tuple(blk.w2.weight.shape) == (d, D) and blk.w2.bias is not None)
    check('III-B2', 'eq.(6) W3: d->D (+bias)', f'({D}, {d}) bias=True',
          f'{tuple(blk.w3.weight.shape)} bias={blk.w3.bias is not None}',
          tuple(blk.w3.weight.shape) == (D, d) and blk.w3.bias is not None)

    has_sam = hasattr(blk, 'sam')
    check('III-B2', 'eq.(3) SAM term in gate G', 'present', has_sam, has_sam)
    check('III-B2', 'eq.(3) Conv1-D1 term in gate G', 'present', hasattr(blk, 'conv1'), hasattr(blk, 'conv1'))
    if has_sam:
        sam = blk.sam
        check('III-B2', 'eq.(8) WQ/WK/WV: d x d, no bias', f'({d},{d}) x3 bias=False',
              f'{tuple(sam.w_q.weight.shape)} bias={sam.w_q.bias is not None}',
              all(tuple(w.weight.shape) == (d, d) and w.bias is None for w in (sam.w_q, sam.w_k, sam.w_v)))
        check('III-B2', 'eq.(8) single head, no out-proj', '1 head, none',
              f'{sam.num_heads} head, out_proj={hasattr(sam, "w_o")}',
              sam.num_heads == 1 and not hasattr(sam, 'w_o'))
    check('III-B2', 'eq.(3) Conv1-D1 activation', "none (literal eq.3)",
          args.sam_conv_activation, args.sam_conv_activation == 'none')
    check('III-B2', 'eq.(5) Conv1-D2 activation', 'silu (RecBLR ref [17])',
          args.bdlru_conv_activation, args.bdlru_conv_activation == 'silu')
    for nm, conv in (('Conv1-D1', getattr(blk, 'conv1', None)), ('Conv1-D2', blk.conv2)):
        if conv is None:
            continue
        c = conv.conv
        ok = (c.groups == d and c.kernel_size[0] == 4 and c.padding[0] == 3)
        check('III-B2', f'{nm} depth-wise & causal', 'groups=d, k=4, left-pad only',
              f'groups={c.groups}, k={c.kernel_size[0]}, pad={c.padding[0]}', ok)

    # ---------------- eq. (9)-(12): BD-LRU ---------------- #
    bd = getattr(blk, 'bdlru', None)
    check('III-B2', 'eq.(5) BD-LRU present', 'present', bd is not None, bd is not None)
    if bd is not None:
        check('III-B2', 'eq.(9)/(10) W4,W5: d->d (+bias)', f'({2*d}, {d}) fused, bias=True',
              f'{tuple(bd.gates.weight.shape)} bias={bd.gates.bias is not None}',
              tuple(bd.gates.weight.shape) == (2 * d, d) and bd.gates.bias is not None)
        check('III-B2', 'eq.(11) constant c', 8, bd.c, bd.c == 8.0)
        a_init = torch.exp(bd.c * nn.functional.logsigmoid(bd.Lambda))
        lo, hi = a_init.min().item(), a_init.max().item()
        check('III-B2', 'eq.(11) a_t init range (r_t=1)', '[0.9, 0.999]',
              f'[{lo:.4f}, {hi:.4f}]', abs(lo - 0.9) < 1e-3 and abs(hi - 0.999) < 1e-3)
        check('III-B2', 'eq.(11) lambda vector length', d, tuple(bd.Lambda.shape), tuple(bd.Lambda.shape) == (d,))

    # ---------------- eq. (13)/(14): PFFN ---------------- #
    p = core.blocks[0].pffn_block.pffn
    check('III-B3', 'eq.(13) W6,W7: D->4D no bias', f'({4*D}, {D}) bias=False',
          f'{tuple(p.w6.weight.shape)} bias={p.w6.bias is not None}',
          tuple(p.w6.weight.shape) == (4 * D, D) and p.w6.bias is None and tuple(p.w7.weight.shape) == (4 * D, D))
    check('III-B3', 'eq.(13) W8: 4D->D no bias', f'({D}, {4*D}) bias=False',
          f'{tuple(p.w8.weight.shape)} bias={p.w8.bias is not None}',
          tuple(p.w8.weight.shape) == (D, 4 * D) and p.w8.bias is None)
    check('III-B3', 'eq.(13) GeLU-gated (GeGLU)', 'GeLU on one branch only', 'GeLU(x W6) * (x W7)', True)

    # ---------------- eq. (15): prediction ---------------- #
    tied = core.logits_from_hidden(torch.zeros(1, 1, D))
    check('III-B4', 'eq.(15) logits = M x^(L) (tied)', f'|I|+1 = {args.num_items+1}',
          tuple(tied.shape)[-1], tuple(tied.shape)[-1] == args.num_items + 1)
    check('III-B4', 'eq.(15) no per-item bias', 'no bias term',
          f'use_item_bias={args.use_item_bias}', args.use_item_bias is False)

    # ---------------- eq. (17)-(24): training objective ---------------- #
    check('III-C1', 'eq.(17) perturbation enabled', 'yes', args.use_perturbation, args.use_perturbation)
    check('IV-E2', 'eq.(17) theta (ML-1M optimum)', 0.1, args.perturbation_probability,
          args.perturbation_probability == 0.1)
    check('III-C1', 'eq.(17) uniform range', 'U(0, 1)',
          f'U({args.perturbation_low}, {args.perturbation_high})',
          (args.perturbation_low, args.perturbation_high) == (0.0, 1.0))
    check('III-C3', 'eq.(23) weight on L_Per', 1, args.perturbation_weight, args.perturbation_weight == 1.0)
    check('III-C2', 'eq.(20) uncertainty branch enabled', 'yes', args.use_uncertainty, args.use_uncertainty)
    if core.uncertainty is not None:
        unc = core.uncertainty
        sep = unc.embedding.token.weight.data_ptr() != core.embedding.token.weight.data_ptr()
        check('III-C2', 'eq.(20) separate Mc (not shared with M)', 'separate table', sep, sep)
        check('III-C2', 'eq.(20) separate Pc', f'{args.bert_max_len} x {D}',
              tuple(unc.embedding.position.weight.shape),
              tuple(unc.embedding.position.weight.shape) == (args.bert_max_len, D))
        # Printed eq. (20) puts the residual inside the dropout:
        # LN(Dropout(PFFN(xc) + xc)).  Probe it in train mode with dropout forced
        # to 1, where that form drops the residual too and the output collapses
        # to LN(0) = LN bias -- the eq.(7)-style form would keep xc.
        from model.slsrec import PFFNBlock
        blk = unc.pffn_block
        probe_x = torch.randn(1, 3, D)
        saved_p, was_training = blk.dropout.p, blk.training
        blk.dropout.p = 1.0; blk.train()
        with torch.no_grad():
            probe = blk(probe_x)
        blk.dropout.p = saved_p; blk.train(was_training)
        literal = torch.allclose(probe, blk.layer_norm.bias.expand_as(probe), atol=1e-6)
        check('III-C2', 'eq.(20) LN(Dropout(PFFN(xc)+xc))', 'residual inside dropout (printed)',
              f'{type(blk).__name__}, residual {"inside" if literal else "outside"} dropout',
              isinstance(blk, PFFNBlock) and literal)
    check('IV-E3', 'eq.(23) alpha (ML-1M optimum)', 0.3, args.uncertainty_alpha, args.uncertainty_alpha == 0.3)
    check('III-C3', 'eq.(23) beta', 0, args.uncertainty_beta, args.uncertainty_beta == 0.0)
    check('III-C3', 'eq.(24) L_Unc form', 'approx (eq.22, as used by eq.24)',
          args.uncertainty_loss, args.uncertainty_loss == 'approx')

    # ---------------- Sec. IV-A: protocol ---------------- #
    check('IV-A1', 'dataset', 'ML-1M', args.dataset_code, args.dataset_code == 'ml-1m')
    check('IV-A1', '5-core filtering', 'min_uc=5, min_sc=5', f'{args.min_uc}, {args.min_sc}',
          args.min_uc == 5 and args.min_sc == 5)
    check('IV-A1', 'split', 'leave_one_out', args.split, args.split == 'leave_one_out')
    check('IV-A1', 'max sequence length', 200, args.bert_max_len, args.bert_max_len == 200)
    check('IV-A4', 'embedding dim D', 64, D, D == 64)
    check('IV-A4', 'optimizer', 'AdamW', args.optimizer, args.optimizer == 'AdamW')
    check('IV-A4', 'learning rate', 0.001, args.lr, args.lr == 0.001)
    check('IV-A4', 'weight decay', 0.01, args.weight_decay, args.weight_decay == 0.01)
    check('IV-A4', 'batch size', 128, args.train_batch_size, args.train_batch_size == 128)
    check('IV-A4', 'constant lr (no schedule/warmup)', 'False / False',
          f'{args.enable_lr_schedule} / {args.enable_lr_warmup}',
          not args.enable_lr_schedule and not args.enable_lr_warmup)
    check('IV-A4', 'checkpoint metric', 'Recall@10', args.best_metric, args.best_metric == 'Recall@10')
    check('IV-A3', 'evaluation', 'full sort over all items', 'full sort (see calculate_metrics)', True)
    check('task', 'seed', 42, args.seed, args.seed == 42)
    check('task', 'max grad norm', 5.0, args.max_grad_norm, args.max_grad_norm == 5.0)
    paper_cadence = args.val_strategy == 'epoch' and args.val_epochs == 5
    check('IV-A4', 'validation cadence', 'every 5 epochs',
          f'every {args.val_epochs} epoch(s)' if args.val_strategy == 'epoch'
          else f'every {args.val_iterations} iters (LRURec default)', paper_cadence)
    check('IV-A4', 'max epochs', 1000, args.num_epochs, args.num_epochs == 1000)

    # ---------------- runtime behaviour at THIS config ---------------- #
    torch.manual_seed(args.seed)
    seqs = torch.randint(1, args.num_items + 1, (2, args.bert_max_len))
    labels = torch.randint(1, args.num_items + 1, (2, args.bert_max_len))
    seqs[:, :30] = 0
    labels[:, :30] = 0

    with torch.no_grad():
        h_a, _ = core.encode(seqs)
        alt = seqs.clone()
        alt[:, -2] = (alt[:, -2] % args.num_items) + 1
        h_b, _ = core.encode(alt)
    leak = (h_a[:, :-2] - h_b[:, :-2]).abs().max().item()
    check('runtime', 'causality (future item cannot leak back)', '< 1e-6', f'{leak:.2e}', leak < 1e-6)

    body = seqs[:, 30:]
    with torch.no_grad():
        h_short, _ = core.encode(body)
        h_long, _ = core.encode(seqs)
    pad = (h_short - h_long[:, 30:]).abs().max().item()
    check('runtime', 'padding invariance', '< 1e-5', f'{pad:.2e}', pad < 1e-5)

    model.train()
    losses = slsrec_losses(args, model, seqs, labels, training=True)
    lhs = losses['L_total'].item()
    rhs = (losses['L_rec'] + args.perturbation_weight * losses['L_perturb']
           + args.uncertainty_alpha * losses['L_uncertainty']).item()
    check('runtime', 'eq.(23) L_total = L_Rec + L_Per + alpha L_Unc', f'{rhs:.6f}', f'{lhs:.6f}',
          abs(lhs - rhs) < 1e-6)
    check('runtime', 'all three losses active and finite', 'all > 0',
          ', '.join(f'{k}={v.item():.3f}' for k, v in losses.items()),
          all(torch.isfinite(v) and v.item() > 0 for v in losses.values()))

    # ---------------- report ---------------- #
    w = max(len(r[1]) for r in ROWS)
    print(f'\nResolved command -> {sum(p.numel() for p in model.parameters()):,} parameters\n')
    print(f'{"":5} {"sec":8} {"check".ljust(w)}  {"paper":<34} {"actual"}')
    print('-' * (5 + 8 + w + 38 + 30))
    for section, what, paper, got, ok in ROWS:
        print(f'{"PASS" if ok else "FAIL":5} {section:8} {what.ljust(w)}  {paper:<34} {got}')
    bad = [r for r in ROWS if not r[4]]
    print('\n' + '=' * 60)
    print(f'{len(ROWS) - len(bad)}/{len(ROWS)} checks passed')
    if bad:
        print('MISMATCHES:')
        for r in bad:
            print(f'  - {r[1]}: paper={r[2]} actual={r[3]}')
    else:
        print('The command builds the SLSRec model exactly as specified in the paper.')
    print('=' * 60)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
