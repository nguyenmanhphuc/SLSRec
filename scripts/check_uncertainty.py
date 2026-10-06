"""Inspect a trained SLSRec uncertainty branch without training or changing weights.

Example (run_config.json supplies the model hyperparameters):
    python scripts/check_uncertainty.py --run-dir experiments/slsrec/RUN_NAME

For a legacy run without run_config.json, specify --dataset-code and the
original --uncertainty-loss / --unc-dropout-form explicitly. Other model
hyperparameters then come from the current dataset defaults.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        help='defaults to RUN_DIR/models/best_acc_model.pth')
    parser.add_argument('--dataset-code', help='required if the run has no saved config')
    parser.add_argument('--dataset-pkl', type=Path, help='override the dataset cache path')
    parser.add_argument('--uncertainty-loss', choices=['approx', 'exact'])
    parser.add_argument('--unc-dropout-form', choices=['fig2', 'eq20'])
    parser.add_argument('--samples', type=int, default=512, help='number of train windows')
    parser.add_argument('--sampling', choices=['random', 'first'], default='random')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train-repeats', type=int, default=3,
                        help='independent dropout passes; 0 skips train-mode probing')
    parser.add_argument('--seed', type=int, default=42, help='diagnostic sampling/dropout seed')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, help='optional JSON report; no files otherwise')
    options = parser.parse_args()
    if min(options.samples, options.batch_size, options.threads) < 1:
        parser.error('--samples, --batch-size and --threads must be positive')
    if options.train_repeats < 0:
        parser.error('--train-repeats must be nonnegative')
    return options


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def resolve_args(options, config, torch):
    config_path = options.run_dir / 'run_config.json'
    saved = json.loads(config_path.read_text(encoding='utf-8'))['hparams'] \
        if config_path.is_file() else {}
    dataset = options.dataset_code or saved.get('dataset_code')
    if not dataset:
        raise ValueError('No saved config: specify --dataset-code and original model settings.')
    args = config.parser.parse_args(['--model_code=slsrec', '--dataset_code=' + dataset])
    config.set_template(args)
    # Overlay AFTER set_template, which would otherwise overwrite recorded settings.
    vars(args).update(saved)
    args.dataset_code = dataset
    notes = []
    if not saved:
        notes.append('No run_config.json: using current dataset defaults for unspecified settings; '
                     'verify these match the training run.')
    if options.uncertainty_loss is not None:
        args.uncertainty_loss = options.uncertainty_loss
    if options.unc_dropout_form is not None:
        args.unc_dropout_form = options.unc_dropout_form
    elif 'unc_dropout_form' not in saved:
        notes.append('Dropout form was not recorded: train-mode probe assumes fig2. '
                     'Pass --unc-dropout-form=eq20 for a run trained with equation (20). '
                     'Eval-mode results do not depend on this choice.')
        args.unc_dropout_form = 'fig2'
    if args.model_code != 'slsrec' or not args.use_uncertainty:
        raise ValueError('This diagnostic requires a SLSRec checkpoint with uncertainty enabled.')
    args.device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if options.device == 'auto' else options.device
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device=cpu.')
    if args.scan_impl != 'torch':
        notes.append('Diagnostic uses the torch scan backend instead of ' + args.scan_impl + '.')
    args.scan_impl = 'torch'
    args.num_workers = 0
    return args, notes


def logged_losses(run_dir, alpha, selected_epoch=None):
    path = run_dir / 'train_losses.jsonl'
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            # A running trainer can be halfway through writing its final line.
    if not records:
        return None
    # Training appends logs when a name is reused. Keep the most recent run.
    session_start = 0
    for index in range(1, len(records)):
        if records[index].get('accum_iter', 0) < records[index - 1].get('accum_iter', 0):
            session_start = index
    records = records[session_start:]

    def summarize(record):
        row = {key: record.get(key) for key in
               ['epoch', 'L_rec', 'L_perturb', 'L_uncertainty', 'L_total']}
        weighted = alpha * record['L_uncertainty']
        row['alpha_times_L_uncertainty'] = weighted
        row['weighted_uncertainty_over_rec_loss'] = weighted / record['L_rec'] \
            if record['L_rec'] else None
        total = record.get('L_total', record.get('loss', 0))
        row['weighted_uncertainty_percent_of_total_loss'] = 100 * weighted / total \
            if total else None
        return row

    selected = [row for row in records if row.get('epoch') == selected_epoch]
    return {'older_session_records_ignored': session_start,
            'latest_session_epochs_logged': len(records),
            'first_epoch': summarize(records[0]),
            'losses_at_selected_checkpoint_epoch': summarize(selected[-1]) if selected else None,
            'last_epochs': [summarize(row) for row in records[-5:]],
            'note': 'Final logged epochs can be later than the selected best checkpoint.'}


def probe(model, loader, args, training, repeats, seed, torch):
    """Weight all summaries by valid positions, matching slsrec_losses' mask."""
    import torch.nn.functional as F

    model.train(training)
    parts = {key: [] for key in ['raw', 'weight', 'rec', 'unc']}
    missing_input = 0
    with torch.inference_mode():
        for repeat in range(repeats):
            torch.manual_seed(seed + repeat)
            for seqs, labels in loader:
                seqs, labels = seqs.to(args.device), labels.to(args.device)
                valid = labels != 0
                if not valid.any():
                    continue
                scores = model(seqs)[0][valid]
                targets = labels[valid]
                rec = F.cross_entropy(scores, targets, reduction='none')
                # Call the estimator directly: model.log_variance already applies clamp.
                raw = model.model.uncertainty(seqs, labels)[valid]
                limit = args.uncertainty_logvar_clamp
                clipped = raw.clamp(-limit, limit) if limit and limit > 0 else raw
                inv_var = torch.exp(-clipped)
                if args.uncertainty_loss == 'approx':
                    unc = inv_var * rec
                else:
                    unc = F.cross_entropy(scores * inv_var.unsqueeze(-1), targets,
                                          reduction='none')
                unc = unc + 0.5 * args.uncertainty_beta * clipped
                for key, value in [('raw', raw), ('weight', args.uncertainty_alpha * inv_var),
                                   ('rec', rec), ('unc', unc)]:
                    if not torch.isfinite(value).all():
                        raise FloatingPointError('Non-finite values in ' + key + ' during probe.')
                    parts[key].append(value.detach().cpu().double())
                missing_input += int(((seqs == 0) & valid).sum())
    if not parts['raw']:
        raise ValueError('No valid target positions in the sampled training windows.')
    values = {key: torch.cat(chunks) for key, chunks in parts.items()}
    raw, weight = values['raw'], values['weight']
    limit = args.uncertainty_logvar_clamp
    q = torch.quantile(raw, torch.tensor([0.1, 0.5, 0.9, 0.99], dtype=raw.dtype))
    l_rec, l_unc = values['rec'].mean().item(), values['unc'].mean().item()
    return {
        'mode': 'train_with_dropout' if training else 'eval_without_dropout',
        'dropout_repeats': repeats if training else 0,
        'valid_positions_including_repeats': len(raw),
        'fraction_valid_targets_with_padding_input': missing_input / len(raw),
        'raw_logvar': {'min': raw.min().item(), 'mean': raw.mean().item(),
                       'p10': q[0].item(), 'p50': q[1].item(), 'p90': q[2].item(),
                       'p99': q[3].item(), 'max': raw.max().item()},
        # Strict comparisons: clamp's derivative is zero beyond its boundaries.
        'fraction_above_upper_clamp': (raw > limit).double().mean().item()
            if limit and limit > 0 else None,
        'fraction_below_lower_clamp': (raw < -limit).double().mean().item()
            if limit and limit > 0 else None,
        'mean_alpha_over_variance': weight.mean().item(),
        'median_alpha_over_variance': weight.median().item(),
        'L_rec': l_rec, 'L_uncertainty': l_unc,
        'alpha_times_L_uncertainty': args.uncertainty_alpha * l_unc,
        'weighted_uncertainty_over_rec_loss': args.uncertainty_alpha * l_unc / l_rec
            if l_rec else None,
    }


def main():
    options = parse_options()
    # config has its own module-level parser; keep this CLI separate.
    sys.argv = [sys.argv[0]]
    import torch
    from torch.utils.data import DataLoader, Subset
    import config
    from dataloader.sas import SASTrainDataset
    from model.slsrec import SLSRec

    torch.set_num_threads(options.threads)
    options.run_dir = options.run_dir.resolve()
    args, notes = resolve_args(options, config, torch)
    checkpoint_path = (options.checkpoint or options.run_dir / 'models/best_acc_model.pth').resolve()
    folder = f'{args.dataset_code}_min_rating{args.min_rating}-min_uc{args.min_uc}' \
             f'-min_sc{args.min_sc}-{args.split}'
    dataset_path = (options.dataset_pkl or ROOT / 'data/preprocessed' / folder / 'dataset.pkl').resolve()
    with dataset_path.open('rb') as stream:
        data = pickle.load(stream)
    args.num_items, args.num_users = len(data['smap']), len(data['umap'])
    dataset = SASTrainDataset(args, data['train'], args.bert_max_len,
                              args.sliding_window_size, None)
    count = min(options.samples, len(dataset))
    indices = list(range(count)) if options.sampling == 'first' else \
        random.Random(options.seed).sample(range(len(dataset)), count)
    loader = DataLoader(Subset(dataset, indices), batch_size=options.batch_size,
                        shuffle=False, num_workers=0)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = SLSRec(args)
    model.load_state_dict(checkpoint[config.STATE_DICT_KEY], strict=True)
    model.to(args.device)
    summary_path = options.run_dir / 'run_summary.json'
    selected_epoch = None
    if summary_path.is_file() and checkpoint_path == (options.run_dir / 'models/best_acc_model.pth').resolve():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        selected_epoch = (summary.get('best_val_at') or {}).get('epoch')
    fields = ['dataset_code', 'bert_dropout', 'unc_dropout_form', 'uncertainty_loss',
              'uncertainty_alpha', 'uncertainty_beta', 'uncertainty_logvar_clamp', 'device']
    report = {
        'run_dir': str(options.run_dir), 'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256(checkpoint_path),
        'selected_epoch_from_run_summary': selected_epoch,
        'dataset': str(dataset_path), 'dataset_sha256': sha256(dataset_path),
        'environment': {'python': sys.version.split()[0], 'torch': torch.__version__},
        'source_sha256': {path: sha256(ROOT / path) for path in
                          ['model/slsrec.py', 'model/bdlru.py', 'trainer/slsrec.py', 'config.py']},
        'hparams': {key: getattr(args, key) for key in fields},
        'sampling': {'method': options.sampling, 'seed': options.seed,
                     'windows': count, 'available_windows': len(dataset)},
        'notes': notes + ['No optimizer steps; weights and training logs are unchanged.',
                          'A weak branch at the final checkpoint does not exclude an effect early in training.',
                          'alpha/variance is a CE weight for approx; exact uses it as a temperature scale.'],
        'logged_losses': logged_losses(options.run_dir, args.uncertainty_alpha, selected_epoch),
        'eval': probe(model, loader, args, False, 1, options.seed, torch),
        'train': probe(model, loader, args, True, options.train_repeats, options.seed, torch)
            if options.train_repeats else None,
    }
    result = json.dumps(report, indent=2, allow_nan=False)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(result + '\n', encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
