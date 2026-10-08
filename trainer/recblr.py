"""RecBLR trainer: plain per-position cross entropy (RecBLR's ``loss_type: CE``).

Evaluation is SLSRecTrainer's (last position, full sort, seen items and padding
masked, held-out CE logged), so RecBLR and every SLSRec variant are scored by
the same code.
"""

from .slsrec import SLSRecTrainer


class RecBLRTrainer(SLSRecTrainer):
    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)[0]
        return self.ce(logits.view(-1, logits.size(-1)), labels.view(-1))
