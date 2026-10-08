"""Read-only checkpoint probe: no optimizer steps or training-log changes."""
import hashlib
import json
import pickle
import random
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.slsrec import SLSRec
from dataloader.sas import SASTrainDataset

torch.set_num_threads(2)
DATA = ROOT / 'data/preprocessed/ml-1m_min_rating0-min_uc5-min_sc5-leave_one_out/dataset.pkl'
with DATA.open('rb') as stream:
    data = pickle.load(stream)
report = {
    'method': 'Eval mode disables dropout. Random 128 train windows seed 42. Output seeds 0,1,2. Gradients use first 16 sampled windows and seeds 0,1. No optimizer steps.',
    'python': sys.version.split()[0], 'torch': torch.__version__,
    'dataset_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
    'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ['model/slsrec.py','model/bdlru.py','trainer/slsrec.py']},
    'limitations': ['Checkpoint-local sensitivity does not prove causality for generalization.', 'Training windows, not held-out evaluation; top10 overlap is a sensitivity metric.', 'Gradient sample has 16 windows; dropout is disabled to isolate perturbation.'],
}

def overlap(clean, dirty, seqs):
    c, p = clean[:, -1].clone(), dirty[:, -1].clone()
    c.scatter_(1, seqs, -1e9); p.scatter_(1, seqs, -1e9)
    c[:, 0] = -1e9; p[:, 0] = -1e9
    ct, pt = c.topk(10).indices, p.topk(10).indices
    return sum(torch.isin(pt[i], ct[i]).float().mean().item() for i in range(len(seqs)))

for run in ['C_perturbation_ml-1m_seed42', 'C_perturbation_theta0_ml-1m_seed42']:
    folder = ROOT / 'experiments/slsrec' / run
    cfg = json.loads((folder / 'run_config.json').read_text())
    args = Namespace(**cfg['hparams']); args.device = 'cpu'; args.scan_impl = 'torch'
    model = SLSRec(args)
    checkpoint = folder / 'models/best_acc_model.pth'
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'], strict=True); model.eval()
    ds = SASTrainDataset(args, data['train'], args.bert_max_len, args.sliding_window_size, None)
    ids = random.Random(42).sample(range(len(ds)), 128)
    batches = list(DataLoader(Subset(ds, ids), batch_size=8))
    entry = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'sampled_train_indices': ids, 'output': {}, 'gradient': {}}
    for theta in [0.1, 1.0]:
        for b in model.model.blocks: b.sg_lru.bdlru.perturb_prob = theta
        rows = []
        for seed in [0, 1, 2]:
            torch.manual_seed(seed)
            accum = dict(n=0, ce_clean=0., ce_dirty=0., kl=0., delta2=0., clean2=0., last_overlap=0., users=0)
            with torch.no_grad():
                for seq, lab in batches:
                    clean, dirty = model(seq, perturb=False)[0], model(seq, perturb=True)[0]
                    valid = lab != 0; c, p, y = clean[valid], dirty[valid], lab[valid]
                    accum['n'] += len(y)
                    accum['ce_clean'] += F.cross_entropy(c, y, reduction='sum').item()
                    accum['ce_dirty'] += F.cross_entropy(p, y, reduction='sum').item()
                    accum['kl'] += F.kl_div(F.log_softmax(p, -1), F.log_softmax(c, -1), log_target=True, reduction='sum').item()
                    accum['delta2'] += (p-c).square().sum().item(); accum['clean2'] += c.square().sum().item()
                    accum['last_overlap'] += overlap(clean, dirty, seq); accum['users'] += len(seq)
            rows.append(dict(seed=seed, valid_positions=accum['n'], ce_clean=accum['ce_clean']/accum['n'], ce_dirty=accum['ce_dirty']/accum['n'], KL_clean_to_dirty=accum['kl']/accum['n'], relative_logits_l2=(accum['delta2']/accum['clean2'])**.5, top10_overlap_last=accum['last_overlap']/accum['users']))
        entry['output'][str(theta)] = rows
    seq = torch.cat([b[0] for b in batches[:2]]); lab = torch.cat([b[1] for b in batches[:2]])
    valid = lab != 0; params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    def grad(pert, seed):
        torch.manual_seed(seed)
        logits = model(seq, perturb=pert)[0]; loss = F.cross_entropy(logits[valid],lab[valid])
        return loss.item(), torch.autograd.grad(loss, [p for n,p in params], allow_unused=True)
    lrec, gc = grad(False, 0)
    groups = {'all': lambda n: True, 'embedding': lambda n: '.embedding.' in n, 'sam': lambda n: '.sam.' in n, 'conv2_bdlru': lambda n: '.conv2.' in n or '.bdlru.' in n, 'w3': lambda n: '.w3.' in n, 'pffn': lambda n: '.pffn_block.' in n}
    for theta in [0.1, 1.0]:
        for b in model.model.blocks: b.sg_lru.bdlru.perturb_prob = theta
        rows=[]
        for seed in [0, 1]:
            lp, gp = grad(True, seed); stats={}
            for group, keep in groups.items():
                pairs=[(a,b) for (n,p),a,b in zip(params,gc,gp) if keep(n) and a is not None and b is not None]
                aa=sum(a.square().sum().double() for a,b in pairs); bb=sum(b.square().sum().double() for a,b in pairs); ab=sum((a*b).sum().double() for a,b in pairs)
                stats[group]=dict(cosine=float(ab/(aa.sqrt()*bb.sqrt()).clamp_min(1e-30)), clean_grad_norm=float(aa.sqrt()), dirty_grad_norm=float(bb.sqrt()))
            rows.append(dict(seed=seed,L_rec=lrec,L_per=lp,valid_positions=int(valid.sum()),groups=stats))
        entry['gradient'][str(theta)]=rows
    report[run]=entry
    print(run, 'output and gradients complete', flush=True)
    if run != 'C_perturbation_ml-1m_seed42': continue
    # Gates compare U(0,1) with the original input at exactly the replaced locations.
    # Every layer uses its own original input in the perturbed pass, controlling
    # for upstream changes in layer 2. These means do not establish a classifier.
    captured={}; handles=[]
    def capture_input(index):
        def hook(module, inputs): captured[index]=inputs[0].detach()
        return hook
    for index,b in enumerate(model.model.blocks):
        b.sg_lru.bdlru.perturb_prob=.1
        handles.append(b.sg_lru.bdlru.register_forward_pre_hook(capture_input(index)))
    gate_stats=[dict(n=0,s=0.,s2=0.,p=0,ip=0.,ic=0.,ap=0.,ac=0.,bp=0.,bc=0.) for _ in model.model.blocks]
    torch.manual_seed(0)
    with torch.no_grad():
        for seq,lab in batches:
            model(seq,perturb=True)
            for index,b in enumerate(model.model.blocks):
                bd=b.sg_lru.bdlru; original=captured[index]; pm=bd.last_perturb_mask; real=seq>0; vals=original[real]; st=gate_stats[index]
                st['n']+=vals.numel();st['s']+=vals.sum().item();st['s2']+=vals.square().sum().item()
                # This gate counterfactual uses fresh iid U(0,1) at the same
                # selected positions; it does not reuse the actual forward noise.
                noise=torch.rand_like(original)
                rc,ic=bd.gates(original).chunk(2,-1);rp,ip=bd.gates(noise).chunk(2,-1)
                rc,ic,rp,ip=rc.sigmoid(),ic.sigmoid(),rp.sigmoid(),ip.sigmoid()
                ac=torch.exp(bd.c*F.logsigmoid(bd.Lambda)*rc);ap=torch.exp(bd.c*F.logsigmoid(bd.Lambda)*rp)
                bc=(1-ac.square()).clamp_min(1e-8).sqrt()*ic*original;bp=(1-ap.square()).clamp_min(1e-8).sqrt()*ip*noise
                count=int(pm.sum());st['p']+=count
                for key,tensor in [('ip',ip),('ic',ic),('ap',ap),('ac',ac)]: st[key]+=tensor[pm].mean(-1).sum().item()
                st['bp']+=bp[pm].norm(dim=-1).sum().item();st['bc']+=bc[pm].norm(dim=-1).sum().item()
    for h in handles:h.remove()
    entry['gates_note']='Compare original pre-replacement input and fresh iid U(0,1) at exactly the sampled replacement locations. Layer 2 pre-replacement inputs include upstream layer 1 perturbation. This is a gate counterfactual, not proof of a noise classifier.'
    entry['gates']=[dict(layer=i,pre_replacement_input_mean=s['s']/s['n'],pre_replacement_input_std=(s['s2']/s['n']-(s['s']/s['n'])**2)**.5,replaced_positions=s['p'],mean_i_uniform=s['ip']/s['p'],mean_i_original_selected=s['ic']/s['p'],mean_a_uniform=s['ap']/s['p'],mean_a_original_selected=s['ac']/s['p'],mean_input_term_norm_uniform=s['bp']/s['p'],mean_input_term_norm_original_selected=s['bc']/s['p']) for i,s in enumerate(gate_stats)]
    # H := 0 sensitivity confirms whether a residual alone carries the checkpoint.
    total=dict(n=0,clean_ce=0.,zeroH_ce=0.,overlap=0.,users=0)
    with torch.no_grad():
        for seq,lab in batches:
            clean=model(seq,perturb=False)[0]
            hs=[b.sg_lru.bdlru.register_forward_hook(lambda m,i,o:torch.zeros_like(o)) for b in model.model.blocks]
            try: dirty=model(seq,perturb=False)[0]
            finally:
                for h in hs:h.remove()
            valid=lab!=0;total['n']+=int(valid.sum())
            total['clean_ce']+=F.cross_entropy(clean[valid],lab[valid],reduction='sum').item()
            total['zeroH_ce']+=F.cross_entropy(dirty[valid],lab[valid],reduction='sum').item()
            total['overlap']+=overlap(clean,dirty,seq);total['users']+=len(seq)
    entry['zero_H']=dict(ce_clean=total['clean_ce']/total['n'],ce_zero_H=total['zeroH_ce']/total['n'],top10_overlap=total['overlap']/total['users'])
    print('gate and zero H complete', entry['gates'], entry['zero_H'], flush=True)

output=Path(__file__).with_name('perturbation_checkpoint_probe.json')
output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('Saved', output)
