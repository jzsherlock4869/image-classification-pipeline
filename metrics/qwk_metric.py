# quadratic weighted kappa metric for classification task

import numpy as np
# from sklearn.metrics import top_k_accuracy_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score

# import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from utils.meter_utils import AverageMeter

class MetricQuadWeightKappa:
    """
        metric quadratic weighted kappa
    """
    def __init__(self) -> None:
        self.lbls = []
        self.preds = []

    def update(self, lbl, pred):
        # lbl: Tensor size [n, 1], e.g. 2 (2-th class)
        # pred: prob vec size [n, c], e.g. [0.1, 0.3, 0.6] (top 1 is 2-th class)
        bs = lbl.size()[0]
        pred_np = pred.detach().cpu().numpy()
        for i in range(bs):
            self.lbls.append(lbl[i].item())
            self.preds.append(pred_np[i])
    
    def num_sample(self):
        return len(self.lbls)

    def reset(self):
        self.lbls = []
        self.preds = []

    def calc(self):
        kappa = cohen_kappa_score(np.array(self.lbls), np.array(self.preds), weights='quadratic')
        return kappa